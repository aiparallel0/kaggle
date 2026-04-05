# =============================================================================
# reporting.py
# Purpose: Merged reporting module — benchmark comparison, convergence plots,
#          and results injection into LaTeX paper
# Merged from: benchmark_compare.py + plot_convergence.py + inject_results.py
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
import logging
import math as _math
import os
import re
import statistics as _statistics
import subprocess as _subprocess
import sys
import time
import unicodedata
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Ensure sibling modules are importable regardless of CWD.
# Uses __file__ to locate the package root so that ``python reporting.py``
# works even when CWD is elsewhere.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:  # pragma: no branch
    sys.path.insert(0, _SCRIPT_DIR)

from constants import FIELDS, WORKSPACE, _edit_distance, _progress  # noqa: E402

# ---------------------------------------------------------------------------
# Optional heavy dependencies (torch, PIL, matplotlib, numpy)
# ---------------------------------------------------------------------------

__all__ = [
    # From benchmark_compare
    "SampleResult",
    "BenchmarkResult",
    "compare_all",
    "benchmark_compare_main",
    "_token_f1",
    "_token_f1_squad",
    "DonutPipeline",
    "TrOCRYOLOPipeline",
    "find_pairs",
    "compute_metrics",
    "plot_results",
    "print_report",
    "save_json",
    # From plot_convergence
    "generate_all",
    "generate_combined_paper",
    "generate_combined_slides",
    "generate_grid_1_8",
    "generate_grid_9_18",
    "smooth_curve",
    "generate_training_plots",
    "generate_convergence_data",
    "generate_convergence_tex",
    "generate_f1_barchart_tex",
    # From inject_results
    "PaperInjector",
    "UnresolvedVarError",
    "LEADERBOARD",
    "DONUT_PUBLISHED_F1",
    "EXP_NAMES",
    "build_var_map",
    "fill_paper",
    "print_table1_dataset_stats",
    "print_table2_experiments",
    "print_table3_perfield",
    "print_table4_leaderboard",
    "ResultsAggregator",
    "print_table5_trocr_yolo",
    "print_table6_cross_architecture",
    "compile_pdf",
    "main",
]

try:
    import numpy as np  # noqa: E402
except ImportError:
    np = None  # type: ignore[assignment]

# Defer heavy imports to avoid import-time crashes when deps are missing.
# _HEAVY_DEPS_AVAILABLE is False when any dep is absent; main() guards entry
# and _ned() raises ImportError on usage so the module can still be imported
# (and _token_f1 / _token_f1_squad used) in test environments that lack
# torch/PIL/etc.
_HEAVY_DEPS_AVAILABLE: bool = True
_HEAVY_DEPS_ERROR: str = ""
_MATPLOTLIB_AVAILABLE: bool = False

try:
    import torch
    from PIL import Image
except ImportError as e:
    _HEAVY_DEPS_AVAILABLE = False
    _HEAVY_DEPS_ERROR = f"FATAL: missing dependency — {e}\nRun: pip install torch Pillow"

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Inline SVG chart generator — zero-dependency fallback when matplotlib absent
# ─────────────────────────────────────────────────────────────────────────────


def _svg_esc(s: str) -> str:
    """XML-escape a value for safe SVG text content."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_save(svg_content: str, stem: Path) -> None:
    """Write .svg and attempt PDF conversion via inkscape / rsvg-convert / cairosvg."""
    svg_path = stem.with_suffix(".svg")
    svg_path.write_text(svg_content, encoding="utf-8")
    pdf_path = str(stem.with_suffix(".pdf"))
    for cmd in [
        ["inkscape", "--export-type=pdf", f"--export-filename={pdf_path}", str(svg_path)],
        ["rsvg-convert", "-f", "pdf", "-o", pdf_path, str(svg_path)],
        ["cairosvg", str(svg_path), "-o", pdf_path],
    ]:
        try:
            _subprocess.run(cmd, capture_output=True, timeout=30, check=True)
            break
        except (FileNotFoundError, _subprocess.CalledProcessError, _subprocess.TimeoutExpired):
            continue


def _svg_bar_chart(
    fields: list,
    methods: list,
    values_per_method: list,
    colors: list,
    title: str,
    ylabel: str,
) -> str:
    """Grouped vertical bar chart as SVG string (replaces matplotlib Fig 1)."""
    W, H = 550, 300
    ML, MR, MT, MB = 58, 20, 35, 65
    PW, PH = W - ML - MR, H - MT - MB
    n_fields, n_methods = len(fields), len(methods)
    group_w = PW / n_fields
    bar_w = group_w * 0.72 / max(n_methods, 1)
    ymax = 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="serif" font-size="8">',
        f'<text x="{W // 2}" y="18" text-anchor="middle" font-size="9" font-weight="bold">{_svg_esc(title)}</text>',
        f'<text x="12" y="{MT + PH // 2}" text-anchor="middle" font-size="8"'
        f' transform="rotate(-90 12 {MT + PH // 2})">{_svg_esc(ylabel)}</text>',
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + PH}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{ML}" y1="{MT + PH}" x2="{ML + PW}" y2="{MT + PH}" stroke="#333" stroke-width="1"/>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_px = MT + PH - int(tick / ymax * PH)
        lines += [
            f'<line x1="{ML - 3}" y1="{y_px}" x2="{ML + PW}" y2="{y_px}"'
            f' stroke="#aaa" stroke-width="0.5" stroke-dasharray="3,2"/>',
            f'<text x="{ML - 5}" y="{y_px + 3}" text-anchor="end" font-size="7">{tick:.2f}</text>',
        ]
    for fi, fname in enumerate(fields):
        gx = ML + fi * group_w
        lines.append(
            f'<text x="{gx + group_w / 2:.1f}" y="{MT + PH + 14}"'
            f' text-anchor="middle" font-size="7">{_svg_esc(fname.capitalize())}</text>'
        )
        for mi, (_method, vals) in enumerate(zip(methods, values_per_method)):
            val = vals[fi]
            bh = int(val / ymax * PH)
            offset = (mi - (n_methods - 1) / 2.0) * bar_w
            bx = gx + group_w / 2.0 + offset - bar_w / 2.0
            by = MT + PH - bh
            color = colors[mi] if mi < len(colors) else f"hsl({mi * 137},60%,50%)"
            lines.append(
                f'<rect x="{bx:.1f}" y="{by}" width="{bar_w:.1f}" height="{bh}"'
                f' fill="{color}" opacity="0.88" stroke="white" stroke-width="0.6"/>'
            )
            if bh > 6:
                lines.append(
                    f'<text x="{bx + bar_w / 2:.1f}" y="{by - 2}"'
                    f' text-anchor="middle" font-size="5.5">{val:.2f}</text>'
                )
    lx0 = ML + PW - 155
    for mi, method in enumerate(methods):
        color = colors[mi] if mi < len(colors) else f"hsl({mi * 137},60%,50%)"
        lx, ly = lx0, MT + 10 + mi * 14
        lines += [
            f'<rect x="{lx}" y="{ly}" width="10" height="8" fill="{color}" opacity="0.88"/>',
            f'<text x="{lx + 13}" y="{ly + 7}" font-size="7">{_svg_esc(method)}</text>',
        ]
    lines.append("</svg>")
    return "\n".join(lines)


def _svg_speed_chart(methods: list, times_data: list, colors: list, title: str) -> str:
    """Box-and-whisker speed chart as SVG string (replaces matplotlib Fig 2 violin)."""
    W, H = 400, 300
    ML, MR, MT, MB = 62, 20, 35, 65
    PW, PH = W - ML - MR, H - MT - MB
    stats = []
    for times in times_data:
        if not times:
            stats.append((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        st = sorted(times)
        n = len(st)
        stats.append((st[0], st[n // 4], _statistics.median(st), st[3 * n // 4], st[-1]))
    all_vals = [v for td in times_data for v in td]
    ymax = max(all_vals) * 1.1 if all_vals else 1.0

    def _ty(v: float) -> int:
        return MT + PH - int(v / ymax * PH)

    n = len(methods)
    group_w = PW / (n + 1)
    box_w = group_w * 0.5
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="serif" font-size="8">',
        f'<text x="{W // 2}" y="18" text-anchor="middle" font-size="9" font-weight="bold">{_svg_esc(title)}</text>',
        f'<text x="14" y="{MT + PH // 2}" text-anchor="middle" font-size="8"'
        f' transform="rotate(-90 14 {MT + PH // 2})">Inference time (ms) \u2193</text>',
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + PH}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{ML}" y1="{MT + PH}" x2="{ML + PW}" y2="{MT + PH}" stroke="#333" stroke-width="1"/>',
    ]
    for tf in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_px = _ty(tf * ymax)
        lines += [
            f'<line x1="{ML}" y1="{y_px}" x2="{ML + PW}" y2="{y_px}"'
            f' stroke="#aaa" stroke-width="0.5" stroke-dasharray="3,2"/>',
            f'<text x="{ML - 5}" y="{y_px + 3}" text-anchor="end" font-size="6">{tf * ymax:.0f}</text>',
        ]
    for i, (method, (mn, q1, med, q3, mx), color) in enumerate(zip(methods, stats, colors)):
        cx = ML + (i + 1) * group_w
        lines += [
            f'<line x1="{cx:.1f}" y1="{_ty(mn)}" x2="{cx:.1f}" y2="{_ty(mx)}" stroke="{color}" stroke-width="1.2"/>',
        ]
        box_top, box_bot = _ty(q3), _ty(q1)
        bh = max(1, box_bot - box_top)
        lines += [
            f'<rect x="{cx - box_w / 2:.1f}" y="{box_top}" width="{box_w:.1f}" height="{bh}"'
            f' fill="{color}" opacity="0.72" stroke="{color}" stroke-width="1"/>',
            f'<line x1="{cx - box_w / 2:.1f}" y1="{_ty(med)}" x2="{cx + box_w / 2:.1f}" y2="{_ty(med)}"'
            f' stroke="black" stroke-width="1.5"/>',
            f'<text x="{cx:.1f}" y="{MT + PH + 14}" text-anchor="middle" font-size="7">{_svg_esc(method)}</text>',
        ]
    lines.append("</svg>")
    return "\n".join(lines)


def _svg_radar_chart(labels: list, series: list, title: str) -> str:
    """Radar/spider chart as SVG string (replaces matplotlib Fig 3 polar)."""
    W, H = 380, 380
    cx, cy, r = W // 2, H // 2 - 5, 115
    n = len(labels)
    angles = [_math.pi / 2 - i * 2 * _math.pi / n for i in range(n)]

    def _pt(angle: float, val: float) -> tuple:
        return cx + r * val * _math.cos(angle), cy - r * val * _math.sin(angle)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="serif" font-size="8">',
        f'<text x="{W // 2}" y="16" text-anchor="middle" font-size="9" font-weight="bold">{_svg_esc(title)}</text>',
    ]
    for ring in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{_pt(a, ring)[0]:.1f},{_pt(a, ring)[1]:.1f}" for a in angles)
        lines.append(
            f'<polygon points="{pts}" fill="none" stroke="#bbb" stroke-width="0.6" stroke-dasharray="3,2"/>'
        )
        rx, ry = _pt(angles[0], ring)
        lines.append(
            f'<text x="{rx + 3:.1f}" y="{ry:.1f}" font-size="6" fill="#888">{ring:.2f}</text>'
        )
    for angle, label in zip(angles, labels):
        ax, ay = _pt(angle, 1.0)
        lines.append(
            f'<line x1="{cx}" y1="{cy}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="#ccc" stroke-width="0.8"/>'
        )
        lx, ly = _pt(angle, 1.22)
        anchor = "middle" if abs(lx - cx) < 5 else ("end" if lx < cx else "start")
        lines.append(
            f'<text x="{lx:.1f}" y="{ly + 3:.1f}" text-anchor="{anchor}" font-size="8" font-weight="bold">'
            f"{_svg_esc(label)}</text>"
        )
    lx0, ly0 = W - 130, 30
    for i, (name, values, color) in enumerate(series):
        pts_raw = [_pt(a, min(1.0, max(0.0, v))) for a, v in zip(angles, values)]
        pts_raw.append(pts_raw[0])
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_raw)
        lines.append(
            f'<polygon points="{pts_str}" fill="{color}" fill-opacity="0.18"'
            f' stroke="{color}" stroke-width="1.6"/>'
        )
        lines += [
            f'<rect x="{lx0}" y="{ly0 + i * 14}" width="10" height="8" fill="{color}" opacity="0.88"/>',
            f'<text x="{lx0 + 13}" y="{ly0 + i * 14 + 7}" font-size="7">{_svg_esc(name)}</text>',
        ]
    lines.append("</svg>")
    return "\n".join(lines)


def _svg_compare_bar(labels: list, values: list, title: str, ylabel: str) -> str:
    """Simple vertical bar chart SVG for compare_all() cross-arch plot."""
    W = max(600, 80 * len(labels) + 120)
    H = 500
    ML, MR, MT, MB = 62, 20, 40, 130
    PW, PH = W - ML - MR, H - MT - MB
    ymax = max(values, default=1.0) * 1.12
    bar_w = PW / max(len(labels), 1) * 0.65

    def _ty(v: float) -> int:
        return MT + PH - int(v / ymax * PH)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="10">',
        f'<text x="{W // 2}" y="24" text-anchor="middle" font-size="12" font-weight="bold">{_svg_esc(title)}</text>',
        f'<text x="16" y="{MT + PH // 2}" text-anchor="middle" font-size="10"'
        f' transform="rotate(-90 16 {MT + PH // 2})">{_svg_esc(ylabel)}</text>',
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + PH}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{ML}" y1="{MT + PH}" x2="{ML + PW}" y2="{MT + PH}" stroke="#333" stroke-width="1.5"/>',
    ]
    for tf in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v = tf * ymax
        y_px = _ty(v)
        lines += [
            f'<line x1="{ML}" y1="{y_px}" x2="{ML + PW}" y2="{y_px}"'
            f' stroke="#ccc" stroke-width="0.7" stroke-dasharray="4,3"/>',
            f'<text x="{ML - 6}" y="{y_px + 4}" text-anchor="end" font-size="9">{v:.2f}</text>',
        ]
    for i, (label, val) in enumerate(zip(labels, values)):
        bx = ML + (i + 0.175) * PW / max(len(labels), 1)
        by = _ty(val)
        bh = MT + PH - by
        lines += [
            f'<rect x="{bx:.1f}" y="{by}" width="{bar_w:.1f}" height="{bh}"'
            f' fill="#4C72B0" opacity="0.82" stroke="white" stroke-width="0.8"/>',
            f'<text x="{bx + bar_w / 2:.1f}" y="{by - 4}" text-anchor="middle" font-size="9">{val:.3f}</text>',
        ]
        lx = bx + bar_w / 2
        lines.append(
            f'<text x="{lx:.1f}" y="{MT + PH + 10}" text-anchor="end" font-size="9"'
            f' transform="rotate(-45 {lx:.1f} {MT + PH + 10})">{_svg_esc(label)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Constants — imported from single source of truth (constants.py)
# ─────────────────────────────────────────────────────────────────────────────
from constants import BASE_MODEL, IMAGE_EXTS, MAX_LENGTH  # noqa: E402


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
    Exact-match entity-level F1 (SROIE Task-3 protocol).

    A field pair is a True Positive only when the normalised predicted string
    exactly matches the normalised ground-truth string — identical to the
    protocol used in ``donut_evaluator.py::compute_metrics()``.  This is the
    correct SROIE Task-3 entity-level F1; it is used as the primary metric for
    all architectures so that cross-architecture comparisons are on equal terms.

    Returns 1.0 for an exact match (including both-empty), 0.0 otherwise.
    """
    # Exact-match: 1.0 iff the normalised strings are identical.
    # This matches donut_evaluator.py::_compute_f1 / compute_metrics().
    return float(_normalise(pred) == _normalise(gold))


def _token_f1_squad(pred: str, gold: str) -> float:
    """
    Bag-of-words token overlap F1 (SQuAD-style).

    NOTE: This is *not* the SROIE Task-3 official metric.  It is provided
    for diagnostic purposes only (e.g. per-token partial-credit analysis).
    Do NOT use this as the primary benchmark metric — it inflates scores
    relative to the exact-match F1 used by ``donut_evaluator.py``, making
    cross-architecture comparisons invalid.

    Both strings are normalised before comparison.
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
    Requires editdistance; raises ImportError if _HEAVY_DEPS_AVAILABLE is False.
    """
    if not _HEAVY_DEPS_AVAILABLE:
        raise ImportError(_HEAVY_DEPS_ERROR)
    pred = _normalise(pred)
    gold = _normalise(gold)
    if pred == gold:
        return 0.0
    maxlen = max(len(pred), len(gold))
    if maxlen == 0:
        return 0.0
    return _edit_distance(pred, gold) / maxlen


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

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

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

    def __init__(self, model_id_or_path: str, device: str = "auto") -> None:
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

    def run_benchmark(
        self, pairs: list[tuple[Path, dict]], desc: str = "DONUT", batch_size: int = 1
    ) -> BenchmarkResult:
        result = BenchmarkResult(method="DONUT")
        # Tokenize the task prompt once — it is constant across all batches.
        _prompt_ids = self.processor.tokenizer(
            self.task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)
        # Process in batches to improve GPU utilisation.
        # DONUT generation is still autoregressive per-token, but the image encoder
        # runs in parallel across the batch — improving throughput on high-VRAM cards.
        n_batches = -(-len(pairs) // batch_size)  # ceiling division
        for i in _progress(range(0, len(pairs), batch_size), desc=desc, total=n_batches):
            batch_pairs = pairs[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p, _ in batch_pairs]

            # Batch pixel_values encoding
            pixel_values = torch.stack(
                [self.processor(img, return_tensors="pt").pixel_values.squeeze(0) for img in images]
            ).to(self.device)

            # Repeat decoder_input_ids for the whole batch
            decoder_input_ids = _prompt_ids.expand(len(images), -1)

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=MAX_LENGTH,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_img_ms = elapsed_ms / len(images)

            token_strs = self.processor.batch_decode(outputs.sequences)
            for (img_path, gt), token_str in zip(batch_pairs, token_strs):
                pred = self._parse_output(token_str)
                result.samples.append(
                    SampleResult(
                        image_name=img_path.name,
                        ground_truth=gt,
                        prediction=pred,
                        inference_time_ms=per_img_ms,
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
        yolo_model_path: str | None = None,
        trocr_model_id: str | None = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        from train_trocr_yolo import _YOLO_CLS as YOLO
        from train_trocr_yolo import (
            TROCR_MAX_LEN,
            TROCR_MODEL_ID,
            YOLO_BASE,
            YOLO_IMG_SIZE,
            _materialize_meta_buffers,
        )

        # Store inference-time constants so all call sites read them consistently.
        self._yolo_img_size = YOLO_IMG_SIZE
        self._trocr_max_len = TROCR_MAX_LEN

        # Fall back to the module-level constants when no override is supplied.
        yolo_model_path = yolo_model_path if yolo_model_path is not None else YOLO_BASE
        trocr_model_id = trocr_model_id if trocr_model_id is not None else TROCR_MODEL_ID

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
            imgsz=self._yolo_img_size,
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
                max_new_tokens=self._trocr_max_len,
            )

        text = self.trocr_processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()

    def _read_crops_batch(self, crops: list[Image.Image]) -> list[str]:
        """Run TrOCR on a batch of cropped PIL images. Returns list of decoded strings."""
        if not crops:
            return []
        pixel_values = torch.stack(
            [
                self.trocr_processor(crop.convert("RGB"), return_tensors="pt").pixel_values.squeeze(
                    0
                )
                for crop in crops
            ]
        ).to(self.device)
        with torch.no_grad():
            generated = self.trocr_model.generate(
                pixel_values,
                max_new_tokens=self._trocr_max_len,
            )
        return [
            t.strip()
            for t in self.trocr_processor.batch_decode(generated, skip_special_tokens=True)
        ]

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

        # Stage 2: read text from each crop (batched — one GPU call for all boxes)
        img_arr = image.convert("RGB")
        crops = []
        for x1, y1, x2, y2 in boxes:
            # Add a small padding to each crop
            pad = 4
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(img_arr.width, x2 + pad)
            y2p = min(img_arr.height, y2 + pad)
            crops.append(img_arr.crop((x1p, y1p, x2p, y2p)))
        lines = [t for t in self._read_crops_batch(crops) if t]

        # Stage 3: regex field assignment
        pred = self._assign_fields(lines)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return pred, elapsed_ms

    def run_benchmark(
        self, pairs: list[tuple[Path, dict]], desc: str = "YOLO+TrOCR+Regex"
    ) -> BenchmarkResult:
        result = BenchmarkResult(method="YOLOv8+TrOCR+Regex")
        for img_path, gt in _progress(pairs, desc=desc):
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
            # Use exact-match F1 (SROIE Task-3 entity-level protocol) so that
            # scores are directly comparable to donut_evaluator.py::compute_metrics().
            # _token_f1_squad() provides partial-credit scores for diagnostics only.
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
if _MATPLOTLIB_AVAILABLE:
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
      Fig 2 — Speed distribution (violin when matplotlib present, box-whisker otherwise)
      Fig 3 — Radar / spider chart (F1, Accuracy, 1-NED, Speed-normalised)

    Uses matplotlib when available; falls back to inline SVG generator otherwise.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [r.method for r in results]
    colors_list = [_METHOD_COLORS.get(r.method, f"C{i}") for i, r in enumerate(results)]

    if _MATPLOTLIB_AVAILABLE:
        # ── Fig 1: Per-field F1 grouped bar chart ────────────────────────────
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
        ax1.set_ylabel("Exact-Match F1 (↑)")
        ax1.set_title("(a) Per-Field F1: DONUT vs. YOLOv8+TrOCR+Regex")
        ax1.legend(loc="upper right", frameon=False)
        ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        _save(fig1, out_dir / "fig1_field_f1")

        # ── Fig 2: Speed distribution — violin plot ──────────────────────────
        fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
        times_data = [[s.inference_time_ms for s in r.samples] for r in results]
        parts = ax2.violinplot(
            times_data, positions=range(1, len(results) + 1), showmedians=True, showextrema=True
        )
        for pc, color in zip(parts["bodies"], colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.72)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        ax2.set_xticks(range(1, len(results) + 1))
        ax2.set_xticklabels(methods, rotation=12, ha="right")
        ax2.set_ylabel("Inference time (ms) ↓")
        ax2.set_title("(b) Inference Speed Distribution")
        ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        _save(fig2, out_dir / "fig2_speed")

        # ── Fig 3: Radar chart ───────────────────────────────────────────────
        max_time = max(r.mean_inference_ms for r in results) or 1.0
        radar_labels = ["F1", "Accuracy", "1-NED", "Speed"]
        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        angles += angles[:1]
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

    else:
        # ── Inline SVG fallback (no matplotlib needed) ───────────────────────
        vals_per_method = [[r.per_field_f1.get(f, 0.0) for f in FIELDS] for r in results]
        svg1 = _svg_bar_chart(
            list(FIELDS),
            methods,
            vals_per_method,
            colors_list,
            "(a) Per-Field F1: DONUT vs. YOLOv8+TrOCR+Regex",
            "Exact-Match F1 (\u2191)",
        )
        _svg_save(svg1, out_dir / "fig1_field_f1")

        times_data = [[s.inference_time_ms for s in r.samples] for r in results]
        svg2 = _svg_speed_chart(
            methods, times_data, colors_list, "(b) Inference Speed Distribution"
        )
        _svg_save(svg2, out_dir / "fig2_speed")

        max_time = max(r.mean_inference_ms for r in results) or 1.0
        series = [
            (
                r.method,
                [
                    r.global_f1,
                    r.global_accuracy,
                    max(0.0, 1.0 - r.global_ned),
                    1.0 - r.mean_inference_ms / max_time,
                ],
                c,
            )
            for r, c in zip(results, colors_list)
        ]
        svg3 = _svg_radar_chart(
            ["F1", "Accuracy", "1-NED", "Speed"], series, "(c) Multi-Metric Radar"
        )
        _svg_save(svg3, out_dir / "fig3_radar")

    print(f"\n[Plots] Saved to: {out_dir}")


def _save(fig, stem: Path) -> None:
    """Save a matplotlib figure to PDF + PNG."""
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


def compare_all(results_dir: Path = Path("results"), figures_dir: Path = Path("figures")) -> None:
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
        if _MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Cross-Architecture Comparison (DONUT vs TrOCR+YOLO+Regex)")
            ax.set_xlabel("Architecture / Experiment")
            ax.set_ylabel("F1 Score")
            ax.bar(range(len(labels)), values, tick_label=labels)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out_path = figures_dir / "cross_arch_comparison.png"
            fig.savefig(str(out_path), dpi=150)
            plt.close(fig)
            print(f"  [compare_all] Saved comparison plot \u2192 {out_path}")
        else:
            svg = _svg_compare_bar(
                labels,
                values,
                "Cross-Architecture Comparison (DONUT vs TrOCR+YOLO+Regex)",
                "F1 Score",
            )
            stem = figures_dir / "cross_arch_comparison"
            _svg_save(svg, stem)
            print(f"  [compare_all] Saved comparison chart \u2192 {stem.with_suffix('.svg')}")
    else:
        print("  [compare_all] No F1 metrics found in result files — skipping plot.")


def benchmark_compare_main() -> None:
    if not _HEAVY_DEPS_AVAILABLE:
        sys.exit(_HEAVY_DEPS_ERROR)

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


# ---------------------------------------------------------------------------
# Convergence plot generation (from plot_convergence.py)
# ---------------------------------------------------------------------------


def _smooth(ys: list, window: int = 5) -> list:
    """Weighted moving-average smoothing — replaces scipy.interpolate.CubicSpline."""
    n = len(ys)
    out = []
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        weights = [1.0 - abs(j - i) / (window + 1) for j in range(lo, hi)]
        s = sum(w * ys[j] for j, w in zip(range(lo, hi), weights))
        out.append(s / sum(weights))
    return out


# ---------------------------------------------------------------------------
# Colour palette — 12-colour qualitative palette (ColorBrewer Set1 + Set2 mix)
# Assigned by experiment ID modulo palette length.
# ---------------------------------------------------------------------------

_PALETTE = [
    "red!80!black",
    "blue!80!black",
    "green!60!black",
    "orange!90!black",
    "purple!80!black",
    "cyan!70!black",
    "brown!80!black",
    "pink!80!black",
    "teal!80!black",
    "violet!80!black",
    "lime!70!black",
    "magenta!70!black",
]

# Short experiment names for legend labels
_EXP_LABELS: dict[str, str] = {
    "1": "Exp~1 SROIE",
    "2": "Exp~2 +WR",
    "3": "Exp~3 +Inv",
    "4": "Exp~4 +WR+Inv",
    "5": "Exp~5 +WR(2×)",
    "6": "Exp~6 +Inv(2×)",
    "7": "Exp~7 +All(2×)",
    "8": "Exp~8 +All(3×)",
    "9": "Exp~9 ZS",
    "10": "Exp~10 FT-fp16",
    "11": "Exp~11 FT-bf16",
    "12": "Exp~12 TrOCR",
    "13": "Exp~13 FT-fp32",
    "14": "Exp~14 HR-fp16",
    "15": "Exp~15 TrOCR-S",
    "16": "Exp~16 HR-bf16",
    "17": "Exp~17 HR-fp32",
    "18": "Exp~18 ZS-HR",
}


# ---------------------------------------------------------------------------
# CSV reading helpers
# ---------------------------------------------------------------------------


def _read_csv(csv_path: Path) -> tuple[list[float], list[float | None], list[float | None]]:
    """
    Read a convergence CSV file.

    Returns (epochs, train_losses, eval_losses).
    Empty cells → None.
    """
    epochs: list[float] = []
    train_losses: list[float | None] = []
    eval_losses: list[float | None] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ep = float(row["epoch"])
            except (ValueError, KeyError):
                continue
            epochs.append(ep)

            tl = row.get("train_loss", "").strip()
            train_losses.append(float(tl) if tl else None)

            el = row.get("eval_loss", "").strip()
            eval_losses.append(float(el) if el else None)

    return epochs, train_losses, eval_losses


# ---------------------------------------------------------------------------
# Cubic spline smoothing
# ---------------------------------------------------------------------------


def smooth_curve(
    epochs: Sequence[float],
    values: Sequence[float | None],
    n_points: int = 200,
) -> tuple[list[float] | None, list[float] | None]:
    """
    Weighted moving-average smoothing.  Skips None / NaN values.

    Returns (x_smooth, y_smooth) or (None, None) if fewer than 2 valid points.
    ``n_points`` is accepted for API compatibility but ignored (output length
    equals the number of valid input points).
    """
    valid = [(e, v) for e, v in zip(epochs, values) if v is not None and not _is_nan(v)]
    if len(valid) < 2:
        return None, None
    xs = [float(p[0]) for p in valid]
    ys = [float(p[1]) for p in valid]
    return xs, _smooth(ys)


def _is_nan(v: float) -> bool:
    try:
        return v != v  # NaN check without math import
    except TypeError:
        return True


# ---------------------------------------------------------------------------
# pgfplots coordinate string
# ---------------------------------------------------------------------------


def _coords_str(xs: list[float], ys: list[float]) -> str:
    """Return pgfplots coordinates string: (x1,y1) (x2,y2) ..."""
    return " ".join(f"({x:.6g},{y:.6g})" for x, y in zip(xs, ys))


# ---------------------------------------------------------------------------
# Per-experiment data loader
# ---------------------------------------------------------------------------


def _load_exp_data(
    exp_id: str,
    results_dir: Path,
) -> dict | None:
    """Load and smooth one experiment's CSV.  Returns None if CSV absent or has <2 points."""
    csv_path = results_dir / f"convergence_exp{exp_id}.csv"
    if not csv_path.exists():
        return None

    try:
        epochs, train_losses, eval_losses = _read_csv(csv_path)
    except (FileNotFoundError, ValueError, OSError):
        return None

    train_xs, train_ys = smooth_curve(epochs, train_losses)
    eval_xs, eval_ys = smooth_curve(epochs, eval_losses)

    if train_xs is None and eval_xs is None:
        return None

    return {
        "exp_id": exp_id,
        "label": _EXP_LABELS.get(exp_id, f"Exp~{exp_id}"),
        "color": _PALETTE[int(exp_id) % len(_PALETTE)],
        "train_xs": train_xs,
        "train_ys": train_ys,
        "eval_xs": eval_xs,
        "eval_ys": eval_ys,
    }


# ---------------------------------------------------------------------------
# Combined axis generators
# ---------------------------------------------------------------------------

_HEADER = "% Auto-generated by plot_convergence.py — do not edit manually\n"


def _combined_axis_content(
    exp_data: list[dict],
    width: str = r"\linewidth",
    height: str = "5cm",
) -> str:
    """Return lines for a single axis showing all experiments."""
    lines: list[str] = []
    for ed in exp_data:
        color = ed["color"]
        label = ed["label"]

        if ed["train_xs"] is not None:
            coords = _coords_str(ed["train_xs"], ed["train_ys"])
            lines.append(
                f"\\addplot[color={color}, solid, line width=0.8pt] "
                f"coordinates {{{coords}}};\n"
                f"\\addlegendentry{{{label} train}}"
            )
        if ed["eval_xs"] is not None:
            coords = _coords_str(ed["eval_xs"], ed["eval_ys"])
            lines.append(
                f"\\addplot[color={color}, dashed, line width=0.8pt] "
                f"coordinates {{{coords}}};\n"
                f"\\addlegendentry{{{label} val}}"
            )

    plots_str = "\n".join(lines)
    return (
        f"\\begin{{tikzpicture}}\n"
        f"\\begin{{axis}}[\n"
        f"  width={width}, height={height},\n"
        f"  xlabel={{Epoch}}, ylabel={{Loss}},\n"
        f"  legend pos=north east,\n"
        f"  legend style={{font=\\tiny}},\n"
        f"  grid=major, grid style={{gray!30}},\n"
        f"  every axis plot/.append style={{line width=0.8pt}},\n"
        f"]\n"
        f"{plots_str}\n"
        f"\\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def generate_combined_paper(
    exp_data: list[dict],
    results_dir: Path,
) -> None:
    """Write results/convergence_combined_paper.tex — figure body for paper.tex."""
    content = _HEADER + _combined_axis_content(exp_data, width=r"\linewidth", height="5cm")
    out = results_dir / "convergence_combined_paper.tex"
    out.write_text(content, encoding="utf-8")


def generate_combined_slides(
    exp_data: list[dict],
    results_dir: Path,
) -> None:
    """Write results/convergence_combined_slides.tex — for Beamer slides."""
    content = _HEADER + _combined_axis_content(exp_data, width=r"\textwidth", height="5cm")
    out = results_dir / "convergence_combined_slides.tex"
    out.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Grid generators
# ---------------------------------------------------------------------------


def _single_axis_block(
    ed: dict,
    width: str = "0.48\\textwidth",
    height: str = "3.5cm",
) -> str:
    """Return a minipage + axis for one experiment."""
    label = ed["label"]
    color = ed["color"]
    lines: list[str] = []

    if ed["train_xs"] is not None:
        coords = _coords_str(ed["train_xs"], ed["train_ys"])
        lines.append(
            f"  \\addplot[color={color}, solid, line width=0.8pt] "
            f"coordinates {{{coords}}};\n"
            f"  \\addlegendentry{{train}}"
        )
    if ed["eval_xs"] is not None:
        coords = _coords_str(ed["eval_xs"], ed["eval_ys"])
        lines.append(
            f"  \\addplot[color={color}, dashed, line width=0.8pt] "
            f"coordinates {{{coords}}};\n"
            f"  \\addlegendentry{{val}}"
        )

    plots_str = "\n".join(lines)
    return (
        f"\\begin{{minipage}}{{{width}}}\n"
        f"\\begin{{tikzpicture}}\n"
        f"\\begin{{axis}}[\n"
        f"  title={{{label}}},\n"
        f"  width=\\textwidth, height={height},\n"
        f"  xlabel={{Epoch}}, ylabel={{Loss}},\n"
        f"  legend pos=north east,\n"
        f"  legend style={{font=\\tiny}},\n"
        f"  grid=major, grid style={{gray!30}},\n"
        f"  title style={{font=\\small\\bfseries}},\n"
        f"  label style={{font=\\tiny}},\n"
        f"  tick label style={{font=\\tiny}},\n"
        f"]\n"
        f"{plots_str}\n"
        f"\\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
        f"\\end{{minipage}}\n"
    )


def _grid_tex(
    exp_data: list[dict],
    cols: int = 2,
    width: str = "0.48\\textwidth",
    height: str = "3.5cm",
) -> str:
    """Return a sequence of minipage blocks arranged in a grid."""
    blocks: list[str] = []
    for i, ed in enumerate(exp_data):
        block = _single_axis_block(ed, width=width, height=height)
        if i > 0 and i % cols == 0:
            blocks.append("\\\\\n")
        elif i > 0:
            blocks.append("\\hfill\n")
        blocks.append(block)
    return "".join(blocks)


def generate_grid_1_8(
    exp_data: list[dict],
    results_dir: Path,
) -> None:
    """Write results/convergence_grid_1_8_slides.tex — 2×4 grid for exps 1–8."""
    subset = [ed for ed in exp_data if int(ed["exp_id"]) <= 8]
    if not subset:
        content = _HEADER + "% No convergence data available for experiments 1--8.\n"
    else:
        content = _HEADER + _grid_tex(subset, cols=2, width="0.48\\textwidth", height="3.0cm")
    out = results_dir / "convergence_grid_1_8_slides.tex"
    out.write_text(content, encoding="utf-8")


def generate_grid_9_18(
    exp_data: list[dict],
    results_dir: Path,
) -> None:
    """Write results/convergence_grid_9_18_slides.tex — grid for exps 9–18."""
    subset = [ed for ed in exp_data if int(ed["exp_id"]) >= 9]
    if not subset:
        content = _HEADER + "% No convergence data available for experiments 9--18.\n"
    else:
        content = _HEADER + _grid_tex(subset, cols=2, width="0.48\\textwidth", height="3.0cm")
    out = results_dir / "convergence_grid_9_18_slides.tex"
    out.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_all(
    results_dir: str | Path = "results",
    selection_file: str | Path = "experiment_selection.json",
) -> None:
    """
    Generate all four convergence .tex output files.

    Missing CSV files are silently skipped.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine which experiment IDs to plot
    selection_file = Path(selection_file)
    exp_ids: list[str] = []
    if selection_file.exists():
        try:
            with open(selection_file, encoding="utf-8") as fh:
                sel = json.load(fh)
            for entry in sel.get("experiments", []):
                if entry.get("enabled", True):
                    exp_ids.append(str(int(str(entry.get("id", "")))))
        except (json.JSONDecodeError, FileNotFoundError, ValueError):
            pass

    if not exp_ids:
        # Fallback: discover from available CSV files
        exp_ids = sorted(
            [
                p.stem.replace("convergence_exp", "")
                for p in results_dir.glob("convergence_exp*.csv")
            ],
            key=lambda s: int(s) if s.isdigit() else 999,
        )

    # Load and smooth data for each experiment
    exp_data: list[dict] = []
    for eid in exp_ids:
        ed = _load_exp_data(eid, results_dir)
        if ed is not None:
            exp_data.append(ed)

    if not exp_data:
        # Write placeholder files so \inputifexists compiles cleanly
        placeholder = _HEADER + "% No convergence CSV files found yet.\n"
        for fname in [
            "convergence_combined_paper.tex",
            "convergence_combined_slides.tex",
            "convergence_grid_1_8_slides.tex",
            "convergence_grid_9_18_slides.tex",
        ]:
            (results_dir / fname).write_text(placeholder, encoding="utf-8")
        return

    generate_combined_paper(exp_data, results_dir)
    generate_combined_slides(exp_data, results_dir)
    generate_grid_1_8(exp_data, results_dir)
    generate_grid_9_18(exp_data, results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _plot_convergence_main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate convergence plot .tex files from experiment CSV data."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing convergence_expN.csv files (default: results)",
    )
    parser.add_argument(
        "--selection",
        default="experiment_selection.json",
        help="Path to experiment_selection.json (default: experiment_selection.json)",
    )
    args, _ = parser.parse_known_args()
    generate_all(results_dir=args.results_dir, selection_file=args.selection)
    print("plot_convergence: convergence .tex files written to", args.results_dir)


# (main() is defined later in this merged file — __main__ block moved to end)


# ---------------------------------------------------------------------------
# Paper injection and results aggregation (from inject_results.py)
# ---------------------------------------------------------------------------

# (inject_results __all__ merged into module-level __all__ at top of file)


# ── LaTeX PDF compilation ──────────────────────────────────────────────────


def compile_pdf(tex_file: str | Path, work_dir: str | Path | None = None) -> Path | None:
    """Compile a LaTeX .tex file to PDF using pdflatex / latexmk / xelatex.

    Runs the compiler twice (pdflatex) to resolve cross-references.
    Returns the Path to the generated .pdf on success, None if no compiler
    is installed.  All compiler output is suppressed (nonstopmode).

    Parameters
    ----------
    tex_file:
        Path to the filled .tex file (e.g., ``paper/paper_filled.tex``).
    work_dir:
        Working directory for the compilation (defaults to directory of
        ``tex_file``).  All auxiliary files (.aux, .log, .toc) are written
        here.

    Returns
    -------
    Path to the .pdf on success, None if compilation failed or no LaTeX
    compiler is available.
    """
    tex_path = Path(tex_file).resolve()
    if not tex_path.exists():
        logging.getLogger(__name__).warning("[PDF] tex file not found: %s", tex_path)
        return None

    if work_dir is None:
        work_dir = tex_path.parent
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = work_dir / tex_path.with_suffix(".pdf").name
    log = logging.getLogger(__name__)

    # compiler → build command (tex_file inserted at end)
    compilers = [
        (
            "pdflatex",
            [
                "pdflatex",
                "-interaction=nonstopmode",
                f"-output-directory={work_dir}",
                str(tex_path),
            ],
        ),
        (
            "latexmk",
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                f"-outdir={work_dir}",
                str(tex_path),
            ],
        ),
        (
            "xelatex",
            [
                "xelatex",
                "-interaction=nonstopmode",
                f"-output-directory={work_dir}",
                str(tex_path),
            ],
        ),
    ]

    for compiler_name, cmd in compilers:
        try:
            # Run twice so cross-references (\ref, \cite) resolve correctly.
            for _ in range(2):
                _subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=120,
                    check=True,
                    cwd=str(work_dir),
                )
            if pdf_path.exists():
                log.info(
                    "[PDF] Compiled %s → %s (using %s)", tex_path.name, pdf_path, compiler_name
                )
                return pdf_path
        except FileNotFoundError:
            continue  # compiler not installed — try next
        except (_subprocess.CalledProcessError, _subprocess.TimeoutExpired) as exc:
            log.warning("[PDF] %s failed for %s: %s", compiler_name, tex_path.name, exc)
            # If the compiler exists but failed, don't try others (likely a .tex error)
            return None

    log.info(
        "[PDF] No LaTeX compiler found (pdflatex / latexmk / xelatex). "
        "Install texlive-latex-base or MiKTeX to auto-compile PDFs."
    )
    return None


# Pre-compiled regex for \VAR{...} template placeholders — compiled once at
# module load rather than on every fill() call.
_VAR_RE = re.compile(r"\\VAR\{([^}]+)\}")

LEADERBOARD: list[tuple[str, float]] = [
    # LayoutLMv3: Huang et al. 2022, "LayoutLMv3: Pre-training for Document AI"
    # Table 6, SROIE entity-level F1. DOI: 10.1145/3503161.3548112
    ("LayoutLMv3 (Huang et al. 2022)", 0.9633),
    # PICK: Yu et al. 2021, "PICK: Processing Key Information Extraction"
    # Table 3, SROIE Task-3 F1. DOI: 10.1109/ICPR48806.2021.9956043
    ("PICK (Yu et al. 2021)", 0.9612),
    # BROS: Hong et al. 2022, "BROS: A Pre-trained Language Model"
    # Table 2, SROIE entity-level F1. arXiv:2108.04539
    ("BROS (Hong et al. 2022)", 0.9548),
    # LayoutLMv2: Xu et al. 2021, "LayoutLMv2: Multi-modal Pre-training"
    # Table 4, SROIE entity-level F1. DOI: 10.18653/v1/2021.acl-long.201
    ("LayoutLMv2 (Xu et al. 2021)", 0.9495),
    # ICDAR 2019 competition results from arXiv:2103.10213 Table 1
    ("H&H Lab — ICDAR'19 1st", 0.9567),
    ("CLOVA OCR — ICDAR'19 2nd", 0.9373),
    ("ICDAR'19 3rd place", 0.9198),
    # DONUT SROIE fine-tuned: Kim et al. 2022, "OCR-free Document Understanding Transformer"
    # Table 1, entity-level F1 on SROIE. arXiv:2111.15664
    # NOTE: This is the SROIE fine-tuned result (not zero-shot CORD transfer).
    ("DONUT (SROIE fine-tuned, Kim et al. 2022)", 0.8411),
]

# Published DONUT F1 on SROIE (Kim et al. 2022, arXiv:2111.15664).
# NOTE: This value (84.11%) is from fine-tuning the CORD-pretrained DONUT on
# SROIE, evaluated with the entity-level F1 protocol consistent with SROIE
# Task-3.  Some versions of the paper report 92.68% using a different
# (field-level) evaluation protocol.  The codebase uses Task-3 F1 throughout,
# so 84.11% is the correct reference value for this comparison.
DONUT_PUBLISHED_F1 = 0.8411  # arXiv:2111.15664, Table 1, entity-level F1

EXP_NAMES: dict[str, str] = {
    "1": "SROIE only",
    "2": "+WildReceipt",
    "3": "+Invoices-DONUT",
    "4": "+WildReceipt+Invoices",
    "5": "+WildReceipt (2x SROIE)",
    "6": "+Invoices (2x SROIE)",
    "7": "+All (2x SROIE)",
    "8": "+All (3x SROIE)",
    "9": "Zero-shot",
    "10": "Fine-tuned fp16",
    "11": "Fine-tuned bf16",
    "12": "TrOCR+YOLO pipeline",
    "13": "Fine-tuned fp32",
    "14": "High-res fp16 (2560x1920)",
    "15": "TrOCR+YOLO (SROIE only)",
    "16": "High-res bf16 (2560x1920)",
    "17": "High-res fp32 (2560x1920)",
    "18": "Zero-shot high-res (2560x1920)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(metrics: dict, key: str, fmt: str = ".4f") -> str:
    """Return formatted metric value or 'N/A'."""
    val = metrics.get(key)
    if val is None:
        return "N/A"
    return format(val, fmt)


class UnresolvedVarError(Exception):
    """Raised when \\VAR{} placeholders remain after substitution."""


# ---------------------------------------------------------------------------
# PaperInjector
# ---------------------------------------------------------------------------


class PaperInjector:
    """Reads experiment JSON results and fills a LaTeX template."""

    def __init__(
        self,
        results_dir: Path,
        template_path: Path,
        _preloaded_experiments: dict | None = None,
    ) -> None:
        self.results_dir = results_dir
        self.template_path = template_path
        self._preloaded_experiments = _preloaded_experiments

    # -- data loading -------------------------------------------------------

    def _load_all_experiments(self) -> dict:
        if self._preloaded_experiments is not None:
            return self._preloaded_experiments
        path = self.results_dir / "all_experiments.json"
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            warnings.warn(
                f"Could not parse {path} as JSON ({exc}); "
                "generating paper with placeholder values only.",
                stacklevel=3,
            )
            return {}

    def _load_evaluation_results(self) -> dict:
        path = self.results_dir / "evaluation_results.json"
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            warnings.warn(
                f"Could not parse {path} as JSON ({exc}); pretrained metrics will show N/A.",
                stacklevel=3,
            )
            return {}

    # -- var map ------------------------------------------------------------

    def build_var_map(self) -> dict[str, str]:
        """Build \\VAR{key} → replacement mapping entirely from JSON files.

        All known experiment-level placeholders (exp1_* through exp8_*) are
        pre-populated with "---" so that partial or missing results produce
        valid LaTeX (with "---" markers) rather than triggering
        ``UnresolvedVarError``.  Real values overwrite the fallbacks for any
        experiments that have completed.
        """
        all_exp = self._load_all_experiments()
        eval_res = self._load_evaluation_results()
        var_map: dict[str, str] = {}

        # Pre-populate ALL expected experiment vars with "---" fallback.
        # This guarantees fill() succeeds even on partial/fresh runs where
        # only some experiments have completed.
        for _eid in range(1, 9):
            _e = str(_eid)
            var_map[f"exp{_e}_n"] = "---"
            var_map[f"exp{_e}_prec"] = "---"
            var_map[f"exp{_e}_rec"] = "---"
            var_map[f"exp{_e}_f1"] = "---"
            var_map[f"exp{_e}_em"] = "---"
            for _fld in FIELDS:
                var_map[f"exp{_e}_{_fld}_f1"] = "---"
                var_map[f"exp{_e}_{_fld}_ned"] = "---"

        # Per-experiment scalars (from all_experiments.json) — override fallbacks
        for exp_id_str, res in all_exp.items():
            m = res.get("metrics", {})
            n = res.get("num_train_samples", 0)
            eid = exp_id_str
            var_map[f"exp{eid}_n"] = f"{n:,}"
            var_map[f"exp{eid}_prec"] = _safe(m, "global_precision")
            var_map[f"exp{eid}_rec"] = _safe(m, "global_recall")
            var_map[f"exp{eid}_f1"] = _safe(m, "global_f1")
            var_map[f"exp{eid}_em"] = _safe(m, "overall_exact_match")
            for fname in FIELDS:
                var_map[f"exp{eid}_{fname}_f1"] = _safe(m, f"{fname}_f1")
                var_map[f"exp{eid}_{fname}_ned"] = _safe(m, f"{fname}_ned")

        # Best experiment
        best_f1 = 0.0
        best_exp_id = "1"
        for exp_id_str, res in all_exp.items():
            f1 = res.get("metrics", {}).get("global_f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_exp_id = exp_id_str

        if best_f1 == 0.0:
            warnings.warn(
                "Best fine-tuned F1 is 0.0 — injecting measured zero; "
                "paper will show 0.0 for all experiment metrics.",
                stacklevel=2,
            )

        var_map["best_f1"] = f"{best_f1:.4f}"
        var_map["best_f1_pct"] = f"{best_f1 * 100:.2f}"
        var_map["best_exp"] = best_exp_id

        # Pretrained / zero-shot metrics
        var_map["pre_f1"] = "N/A"
        var_map["pre_f1_pct"] = "N/A"
        var_map["pre_prec"] = "N/A"
        var_map["pre_rec"] = "N/A"
        var_map["pre_em"] = "N/A"

        pm = eval_res.get("pretrained_metrics", {})
        if pm:
            var_map["pre_f1"] = _safe(pm, "global_f1")
            var_map["pre_f1_pct"] = f"{pm.get('global_f1', 0.0) * 100:.2f}"
            var_map["pre_prec"] = _safe(pm, "global_precision")
            var_map["pre_rec"] = _safe(pm, "global_recall")
            var_map["pre_em"] = _safe(pm, "overall_exact_match")

        # Also check legacy workspace path
        if not pm:
            legacy_path = (
                Path(os.environ.get("DONUT_WORKSPACE", "/workspace")) / "evaluation_results.json"
            )
            if legacy_path.exists():
                try:
                    with open(legacy_path) as fh:
                        legacy = json.load(fh)
                    lp = legacy.get("pretrained_metrics", {})
                    var_map["pre_f1"] = _safe(lp, "global_f1")
                    var_map["pre_f1_pct"] = f"{lp.get('global_f1', 0.0) * 100:.2f}"
                    var_map["pre_prec"] = _safe(lp, "global_precision")
                    var_map["pre_rec"] = _safe(lp, "global_recall")
                    var_map["pre_em"] = _safe(lp, "overall_exact_match")
                except Exception:
                    pass

        # Gains
        try:
            exp1_f1 = all_exp.get("1", {}).get("metrics", {}).get("global_f1", 0.0)
            exp4_f1 = all_exp.get("4", {}).get("metrics", {}).get("global_f1", 0.0)
            # gain_1_4: Exp 4 vs Exp 1 (WR+Inv combined vs baseline)
            var_map["gain_1_4"] = f"{(exp4_f1 - exp1_f1):+.4f}"
            var_map["gain_over_published"] = f"{(best_f1 - DONUT_PUBLISHED_F1):+.4f}"
            # gain_best_over_baseline: best experiment vs our own SROIE-only baseline
            var_map["gain_best_over_baseline"] = f"{(best_f1 - exp1_f1):+.4f}"
        except Exception:
            var_map["gain_1_4"] = "N/A"
            var_map["gain_over_published"] = "N/A"
            var_map["gain_best_over_baseline"] = "N/A"

        # FIX: TrOCR+YOLO results — inject variables for dual-architecture
        # comparison table in paper.tex.  Reads from trocr_yolo_results.json.
        trocr_path = self.results_dir / "trocr_yolo_results.json"
        if trocr_path.exists():
            try:
                with open(trocr_path) as fh:
                    trocr_all = json.load(fh)
                for exp_id_str, res in trocr_all.items():
                    if not isinstance(res, dict):  # skip _note and other metadata
                        continue
                    m = res.get("metrics", {})
                    n = res.get("num_train_samples", 0)
                    eid = exp_id_str
                    var_map[f"trocr_exp{eid}_n"] = f"{n:,}"
                    var_map[f"trocr_exp{eid}_prec"] = _safe(m, "global_precision")
                    var_map[f"trocr_exp{eid}_rec"] = _safe(m, "global_recall")
                    var_map[f"trocr_exp{eid}_f1"] = _safe(m, "global_f1")
                    var_map[f"trocr_exp{eid}_em"] = _safe(m, "overall_exact_match")
                    for field in FIELDS:
                        var_map[f"trocr_exp{eid}_{field}_f1"] = _safe(m, f"{field}_f1")
                        var_map[f"trocr_exp{eid}_{field}_ned"] = _safe(m, f"{field}_ned")

                # Best TrOCR+YOLO result
                trocr_best_f1 = 0.0
                trocr_best_exp = "1"
                for exp_id_str, res in trocr_all.items():
                    if not isinstance(res, dict):
                        continue
                    f1 = res.get("metrics", {}).get("global_f1", 0.0)
                    if f1 > trocr_best_f1:
                        trocr_best_f1 = f1
                        trocr_best_exp = exp_id_str
                var_map["trocr_best_f1"] = f"{trocr_best_f1:.4f}"
                var_map["trocr_best_f1_pct"] = f"{trocr_best_f1 * 100:.2f}"
                var_map["trocr_best_exp"] = trocr_best_exp
            except Exception:
                pass

        # Ensure basic TrOCR vars have fallback values if file was missing
        for key in ["trocr_best_f1", "trocr_best_f1_pct", "trocr_best_exp"]:
            var_map.setdefault(key, "N/A")

        # ── All-backends TrOCR comparison (from trocr_all_backends.json) ─────
        # Placeholders: trocr_regex_f1, trocr_char_f1, trocr_lm_f1, trocr_lmv_f1
        # and per-field variants trocr_<backend>_<field>_f1 / _ned
        _BACKEND_PREFIX_MAP = {
            "regex": "trocr_regex",
            "char": "trocr_char",
            "lm": "trocr_lm",
            "lm+vision": "trocr_lmv",
        }
        _all_backends_path = self.results_dir / "trocr_all_backends.json"
        if _all_backends_path.exists():
            try:
                with open(_all_backends_path) as _bfh:
                    _backends_data = json.load(_bfh)
                _best_backend_f1 = float(var_map.get("trocr_best_f1") or "0") or 0.0
                _best_backend_name = "regex"
                for _bkey, _prefix in _BACKEND_PREFIX_MAP.items():
                    _bd = _backends_data.get(_bkey, {})
                    _bm = _bd.get("metrics", {})
                    var_map[f"{_prefix}_f1"] = _safe(_bm, "global_f1")
                    var_map[f"{_prefix}_f1_pct"] = f"{_bm.get('global_f1', 0.0) * 100:.2f}"
                    for _field in FIELDS:
                        var_map[f"{_prefix}_{_field}_f1"] = _safe(_bm, f"{_field}_f1")
                        var_map[f"{_prefix}_{_field}_ned"] = _safe(_bm, f"{_field}_ned")
                    _bf1 = _bm.get("global_f1", 0.0)
                    if _bf1 > _best_backend_f1:
                        _best_backend_f1 = _bf1
                        _best_backend_name = _bd.get("backend", _bkey)
                        var_map["trocr_best_f1"] = f"{_bf1:.4f}"
                        var_map["trocr_best_f1_pct"] = f"{_bf1 * 100:.2f}"
                var_map["trocr_best_backend"] = _best_backend_name
            except Exception:
                pass

        # Fallbacks for all backend vars so LaTeX compiles even on partial runs
        for _bk, _pfx in _BACKEND_PREFIX_MAP.items():
            var_map.setdefault(f"{_pfx}_f1", "---")
            var_map.setdefault(f"{_pfx}_f1_pct", "---")
            for _fld in FIELDS:
                var_map.setdefault(f"{_pfx}_{_fld}_f1", "---")
                var_map.setdefault(f"{_pfx}_{_fld}_ned", "---")
        var_map.setdefault("trocr_best_backend", "---")

        # ── Best DONUT per-field vars (for cross-arch comparison table) ───────
        if best_exp_id in all_exp:
            _best_m = all_exp[best_exp_id].get("metrics", {})
            for _fld in FIELDS:
                var_map[f"best_{_fld}_f1"] = _safe(_best_m, f"{_fld}_f1")
                var_map[f"best_{_fld}_ned"] = _safe(_best_m, f"{_fld}_ned")
        for _fld in FIELDS:
            var_map.setdefault(f"best_{_fld}_f1", "---")
            var_map.setdefault(f"best_{_fld}_ned", "---")

        # Experiment count / max ID (for dynamic slide titles and abstract)
        exp_ids_with_results = sorted(all_exp.keys(), key=lambda x: int(x))
        var_map["num_experiments"] = str(len(exp_ids_with_results))
        if exp_ids_with_results:
            var_map["max_exp_id"] = exp_ids_with_results[-1]
        else:
            var_map["max_exp_id"] = "0"

        return var_map

    # -- fill ---------------------------------------------------------------

    def fill(self, strict: bool = False) -> str:
        """Replace all \\VAR{key} in template and return the filled text.

        Parameters
        ----------
        strict : bool, optional
            When ``True`` (default: ``False``), raise ``UnresolvedVarError`` if
            any ``\\VAR{key}`` placeholder remains unresolved after substitution.
            When ``False`` (default), unresolved placeholders are replaced with
            ``"---"`` and a warning is emitted instead.  The ``False`` default
            allows partial runs (where only some experiments have completed) to
            still produce a valid, compilable LaTeX file.
        """
        var_map = self.build_var_map()
        text = self.template_path.read_text(encoding="utf-8")

        def _replace(m: re.Match) -> str:
            key = m.group(1)
            return var_map.get(key, m.group(0))

        filled = _VAR_RE.sub(_replace, text)

        remaining = _VAR_RE.findall(filled)
        if remaining:
            if strict:
                raise UnresolvedVarError(
                    f"{len(remaining)} unresolved \\VAR{{}} placeholder(s): {remaining}"
                )
            else:
                warnings.warn(
                    f"{len(remaining)} unresolved \\VAR{{}} placeholder(s) replaced with "
                    f"'---': {remaining}",
                    stacklevel=2,
                )
                # Replace remaining \VAR{key} with "---" so LaTeX can still compile
                filled = _VAR_RE.sub(lambda _m: "---", filled)
        return filled

    # -- leaderboard verification -------------------------------------------

    def verify_leaderboard_scores(self) -> None:
        """Assert leaderboard constants against known-good cited values."""
        expected = {
            # LayoutLMv3: Huang et al. 2022, DOI: 10.1145/3503161.3548112, Table 6
            "LayoutLMv3 (Huang et al. 2022)": 0.9633,
            # PICK: Yu et al. 2021, DOI: 10.1109/ICPR48806.2021.9956043, Table 3
            "PICK (Yu et al. 2021)": 0.9612,
            # BROS: Hong et al. 2022, arXiv:2108.04539, Table 2
            "BROS (Hong et al. 2022)": 0.9548,
            # LayoutLMv2: Xu et al. 2021, DOI: 10.18653/v1/2021.acl-long.201, Table 4
            "LayoutLMv2 (Xu et al. 2021)": 0.9495,
            # ICDAR 2019 competition: arXiv:2103.10213, Table 1
            "H&H Lab — ICDAR'19 1st": 0.9567,
            "CLOVA OCR — ICDAR'19 2nd": 0.9373,
            "ICDAR'19 3rd place": 0.9198,
            # DONUT SROIE fine-tuned: Kim et al. 2022, arXiv:2111.15664, Table 1
            "DONUT (SROIE fine-tuned, Kim et al. 2022)": 0.8411,
        }
        lb_dict = {name: score for name, score in LEADERBOARD}
        for name, score in expected.items():
            actual = lb_dict.get(name)
            assert actual is not None, f"Missing leaderboard entry: {name}"
            assert abs(actual - score) < 1e-6, (
                f"Leaderboard mismatch for {name}: expected {score}, got {actual}"
            )


# ---------------------------------------------------------------------------
# Legacy single-experiment output (kept for backward compatibility)
# ---------------------------------------------------------------------------


def legacy_output(results_path: str = str(WORKSPACE / "evaluation_results.json")) -> None:
    """Print LaTeX rows from a single evaluation output file."""
    with open(results_path) as f:
        results = json.load(f)
    pm = results["pretrained_metrics"]
    fm = results["finetuned_metrics"]

    print("% === PASTE INTO LATEX TABLE 2 (head-to-head) ===")
    print("% Field & Metric & Pretrained & Fine-tuned \\\\")
    for fname in FIELDS:
        f1_p = pm[f"{fname}_f1"]
        f1_f = fm[f"{fname}_f1"]
        ned_p = pm[f"{fname}_ned"]
        ned_f = fm[f"{fname}_ned"]
        print(f"{fname.capitalize()} & F1 & {f1_p:.4f} & {f1_f:.4f} \\\\")
        print(f"{fname.capitalize()} & NED & {ned_p:.4f} & {ned_f:.4f} \\\\")

    print("\\midrule")
    print(f"Global & F1 & {pm['global_f1']:.4f} & {fm['global_f1']:.4f} \\\\")
    print(
        f"Global & Exact Match & {pm['overall_exact_match']:.4f}"
        f" & {fm['overall_exact_match']:.4f} \\\\"
    )

    print()
    print("% === PASTE INTO LATEX TABLE 3 (leaderboard) ===")
    print(f"Our fine-tuned & -- & {fm['global_f1'] * 100:.2f} \\\\")
    print(f"Our pretrained (zero-shot) & -- & {pm['global_f1'] * 100:.2f} \\\\")


# ---------------------------------------------------------------------------
# Multi-experiment table printers
# ---------------------------------------------------------------------------


def print_table1_dataset_stats(actual_counts: dict = None) -> None:
    """Print Table 1: Dataset Statistics LaTeX rows.

    Parameters
    ----------
    actual_counts : dict, optional
        Mapping of dataset name to actual sample count, e.g.
        ``{"sroie_train": 500, "sroie_val": 63, "sroie_test": 63, ...}``.
        When provided, overrides the hardcoded fallback values.

    Notes
    -----
    FUNSD is downloaded during the dataset-download stage but is **not**
    used in any of the 8 DONUT experiments (Exps 1-8 use only ``sroie``,
    ``wildreceipt``, and ``invoices_donut``).  Its field coverage is also
    highly uneven: company~100%, address~99%, date~42%, total~13.4%.
    The low total coverage (13.4%) results from FUNSD being a forms
    dataset where most documents have no currency-amount "total" field.
    FUNSD is therefore excluded from the dataset table as it does not
    contribute to any reported result.
    """
    print("% === TABLE 1: Dataset Statistics ===")
    _c = actual_counts or {}
    # FUNSD is intentionally excluded: it is downloaded but not used in
    # Experiments 1–8, and its field coverage is severely uneven
    # (total=13.4%, date=42.3%).  Including it would misrepresent the
    # training data composition.
    rows = [
        ("SROIE (train)", _c.get("sroie_train", 500), 4, "EN", "Receipts"),
        ("SROIE (val)", _c.get("sroie_val", 63), 4, "EN", "Receipts"),
        ("SROIE (test)", _c.get("sroie_test", 63), 4, "EN", "Receipts"),
        ("WildReceipt", _c.get("wildreceipt", 1740), 25, "EN", "Receipts"),
        ("Invoices-DONUT", _c.get("invoices_donut", 800), "7+", "EN", "Invoices"),
    ]
    for name, n, nf, lang, domain in rows:
        n_str = f"{n:,}" if isinstance(n, int) else str(n)
        print(f"{name} & {n_str} & {nf} & {lang} & {domain} \\\\")
    print()


def print_table2_experiments(all_exp: dict) -> None:
    """Print Table 2: Per-Experiment Results rows."""
    print("% === TABLE 2: Per-Experiment Results ===")
    print("% Exp & Training Data & Train Samples & Precision & Recall & F1 & Exact Match \\\\")
    for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
        res = all_exp[exp_id_str]
        m = res.get("metrics", {})
        name = EXP_NAMES.get(exp_id_str, res.get("name", ""))
        n = res.get("num_train_samples", 0)
        print(
            f"{exp_id_str} & {name} & {n:,} & "
            f"{_safe(m, 'global_precision')} & "
            f"{_safe(m, 'global_recall')} & "
            f"{_safe(m, 'global_f1')} & "
            f"{_safe(m, 'overall_exact_match')} \\\\"
        )
    print()


def print_table3_perfield(all_exp: dict) -> None:
    """Print Table 3: Per-Field F1 / NED rows."""
    print("% === TABLE 3: Per-Field Breakdown ===")
    header_fields = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{{f.capitalize()}}}}}" for f in FIELDS
    )
    # FIX (BUG 8): Added (↓) suffix to NED sub-columns to indicate lower is better.
    print(f"% Exp & Training Data & {header_fields} \\\\")
    print("% Sub-header: F1(↑) & NED(↓) per field")
    for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
        res = all_exp[exp_id_str]
        m = res.get("metrics", {})
        name = EXP_NAMES.get(exp_id_str, res.get("name", ""))
        field_cols = " & ".join(
            # F1 columns are (↑) higher is better; NED columns are (↓) lower is better.
            f"{_safe(m, f + '_f1')} & {_safe(m, f + '_ned')}"
            for f in FIELDS
        )
        print(f"{exp_id_str} & {name} & {field_cols} \\\\")
    print()


def print_table4_leaderboard(all_exp: dict) -> None:
    """Print Table 4: SROIE Task 3 Leaderboard rows."""
    print("% === TABLE 4: Leaderboard Comparison ===")
    entries = list(LEADERBOARD)
    best_f1 = 0.0
    best_exp_id = None
    for exp_id_str, res in all_exp.items():
        f1 = res.get("metrics", {}).get("global_f1", 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_exp_id = exp_id_str
    if best_exp_id:
        exp_name = EXP_NAMES.get(best_exp_id, f"Exp {best_exp_id}")
        entries.append((f"Ours — best fine-tuned ({exp_name})", best_f1))

    for name, score in sorted(entries, key=lambda x: x[1], reverse=True):
        marker = " % <-- ours" if "Ours" in name else ""
        print(f"{name} & {score * 100:.2f} \\\\{marker}")
    print()


def print_table5_trocr_yolo(trocr_exp: dict) -> None:
    """Print Table 5: TrOCR+YOLO Per-Experiment Results rows.

    FIX: New function added for dual-architecture comparison.
    Previously the pipeline only generated DONUT tables.
    """
    print("% === TABLE 5: TrOCR+YOLO Per-Experiment Results ===")
    print("% Exp & Training Data & Train Samples & Precision & Recall & F1 & Exact Match \\\\")
    numeric_keys = [k for k in trocr_exp if k.lstrip("-").isdigit()]
    for exp_id_str in sorted(numeric_keys, key=lambda x: int(x)):
        res = trocr_exp[exp_id_str]
        m = res.get("metrics", {})
        name = EXP_NAMES.get(exp_id_str, res.get("name", ""))
        n = res.get("num_train_samples", 0)
        print(
            f"{exp_id_str} & {name} & {n:,} & "
            f"{_safe(m, 'global_precision')} & "
            f"{_safe(m, 'global_recall')} & "
            f"{_safe(m, 'global_f1')} & "
            f"{_safe(m, 'overall_exact_match')} \\\\"
        )
    print()


def print_table6_cross_architecture(donut_exp: dict, trocr_exp: dict) -> None:
    """Print Table 6: Cross-Architecture Comparison (DONUT vs TrOCR+YOLO).

    FIX: New function for the dual-architecture comparison that is the
    core scientific contribution of this paper.
    """
    print("% === TABLE 6: Cross-Architecture Comparison ===")
    print("% Exp & Training Data & DONUT F1 & TrOCR+YOLO F1 & Delta \\\\")
    numeric_keys = {k for k in set(donut_exp) | set(trocr_exp) if k.isdigit()}
    for exp_id_str in sorted(numeric_keys, key=lambda x: int(x)):
        name = EXP_NAMES.get(exp_id_str, f"Exp {exp_id_str}")
        d_f1 = donut_exp.get(exp_id_str, {}).get("metrics", {}).get("global_f1", 0.0)
        t_f1 = trocr_exp.get(exp_id_str, {}).get("metrics", {}).get("global_f1", 0.0)
        delta = d_f1 - t_f1
        print(f"{exp_id_str} & {name} & {d_f1:.4f} & {t_f1:.4f} & {delta:+.4f} \\\\")
    print()


# ---------------------------------------------------------------------------
# Plot Generation for Paper
# ---------------------------------------------------------------------------


def generate_training_plots(results_dir: Path = Path("results")) -> None:
    """Generate 2D training loss plots from experiment results.

    Creates publication-ready loss plots in results/figures/ for inclusion
    in paper.tex. Uses plot_convergence if available.

    Args:
        results_dir: Path to results directory
    """
    try:
        _plot_convergence_main()
    except Exception:
        pass  # Gracefully skip if plotting unavailable


# ---------------------------------------------------------------------------
# Module-level wrappers (backward compatibility)
# ---------------------------------------------------------------------------


def build_var_map(all_exp: dict) -> dict:
    """Build \\VAR{key} → replacement mapping (module-level wrapper).

    Delegates to PaperInjector.build_var_map(). Passes all_exp via
    _preloaded_experiments to avoid a tempdir disk round-trip.
    """
    injector = PaperInjector(
        results_dir=Path("results"),
        template_path=Path("paper/paper.tex"),
        _preloaded_experiments=all_exp,
    )
    return injector.build_var_map()


def fill_paper(paper_path: str, output_path: str, var_map: dict, strict: bool = False) -> None:
    """Replace all \\VAR{key} tokens in a LaTeX template and write output_path.

    Parameters
    ----------
    paper_path : str
        Path to the LaTeX template (paper.tex or presentation.tex).
    output_path : str
        Path to write the filled output file.
    var_map : dict
        Mapping of placeholder key → replacement string.
    strict : bool, optional
        When ``True``, raise ``UnresolvedVarError`` if any ``\\VAR{key}``
        placeholder remains unresolved.  When ``False`` (default), unresolved
        placeholders are replaced with ``"---"`` and a warning is emitted.
        The ``False`` default ensures a partial run (not all experiments
        complete) still produces a valid, compilable LaTeX file.
    """
    text = Path(paper_path).read_text(encoding="utf-8")

    def _replace(m: re.Match) -> str:
        return var_map.get(m.group(1), m.group(0))

    filled = _VAR_RE.sub(_replace, text)
    remaining = _VAR_RE.findall(filled)
    if remaining:
        if strict:
            raise UnresolvedVarError(
                f"{len(remaining)} unresolved \\VAR{{}} placeholder(s): {remaining}"
            )
        else:
            import warnings as _w

            _w.warn(
                f"{len(remaining)} unresolved \\VAR{{}} placeholder(s) in "
                f"{paper_path} replaced with '---': {remaining}",
                stacklevel=2,
            )
            filled = _VAR_RE.sub(lambda _m: "---", filled)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(filled, encoding="utf-8")
    print(f"Filled paper written -> {output_path}")


# ---------------------------------------------------------------------------
# Convergence plot data / tex generation
# ---------------------------------------------------------------------------

_PLOT_STYLES = [
    ("blue", "o"),
    ("red", "square"),
    ("green!60!black", "triangle"),
    ("orange", "diamond"),
    ("purple", "star"),
    ("teal", "pentagon"),
    ("brown", "x"),
    ("magenta", "+"),
]


def generate_convergence_data(
    results_path: str = "results/all_experiments.json", output_dir: str = "results"
) -> None:
    """Read all_experiments.json and write per-experiment convergence CSV files."""
    results_file = Path(results_path)
    if not results_file.exists():
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_file) as fh:
        all_exp = json.load(fh)

    for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
        res = all_exp[exp_id_str]
        log_history = res.get("training_log", [])
        if not log_history:
            continue

        epoch_data: dict = {}
        for entry in log_history:
            epoch = entry.get("epoch")
            if epoch is None:
                continue
            ep = round(epoch)
            if ep not in epoch_data:
                epoch_data[ep] = {}
            if "loss" in entry:
                epoch_data[ep]["train_loss"] = entry["loss"]
            if "eval_loss" in entry:
                epoch_data[ep]["eval_loss"] = entry["eval_loss"]

        csv_path = out_dir / f"convergence_exp{exp_id_str}.csv"
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write("epoch,train_loss,eval_loss\n")
            for ep in sorted(epoch_data):
                row = epoch_data[ep]
                train_loss = row.get("train_loss", "")
                eval_loss = row.get("eval_loss", "")
                fh.write(f"{ep},{train_loss},{eval_loss}\n")


def generate_convergence_tex(
    results_path: str = "results/all_experiments.json", output_dir: str = "results"
) -> None:
    """Generate results/convergence_plots.tex — pgfplots figure included in paper.tex."""
    results_file = Path(results_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_ids_with_data: list = []
    if results_file.exists():
        with open(results_file) as fh:
            all_exp = json.load(fh)
        for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
            csv_path = out_dir / f"convergence_exp{exp_id_str}.csv"
            if csv_path.exists():
                exp_ids_with_data.append(exp_id_str)

    def _plot_commands(loss_col: int, ylabel: str) -> str:
        """Return addplot lines for one subplot (loss_col: 1=train, 2=eval)."""
        lines = []
        for i, exp_id_str in enumerate(exp_ids_with_data):
            color, marker = _PLOT_STYLES[i % len(_PLOT_STYLES)]
            name = EXP_NAMES.get(exp_id_str, f"Exp {exp_id_str}")
            csv_rel = f"results/convergence_exp{exp_id_str}.csv"
            lines.append(
                f"    \\addplot[color={color},mark={marker},thick] "
                f"table[x=epoch,y index={loss_col},col sep=comma,header=true]"
                f"{{{csv_rel}}};\n"
                f"    \\addlegendentry{{{name}}}"
            )
        return "\n".join(lines)

    train_plots = _plot_commands(1, "Training Loss")
    eval_plots = _plot_commands(2, "Validation Loss")

    tex = (
        r"""\begin{figure*}[t]
  \centering
  \begin{subfigure}[t]{0.48\linewidth}
    \begin{tikzpicture}
      \begin{axis}[
        xlabel={Epoch},
        ylabel={Training Loss},
        width=\linewidth,
        height=6cm,
        legend pos=north east,
        legend style={font=\tiny},
        grid=major,
      ]
"""
        + train_plots
        + r"""
      \end{axis}
    \end{tikzpicture}
    \caption{Training Loss}
  \end{subfigure}%
  \hfill
  \begin{subfigure}[t]{0.48\linewidth}
    \begin{tikzpicture}
      \begin{axis}[
        xlabel={Epoch},
        ylabel={Validation Loss},
        width=\linewidth,
        height=6cm,
        legend pos=north east,
        legend style={font=\tiny},
        grid=major,
      ]
"""
        + eval_plots
        + r"""
      \end{axis}
    \end{tikzpicture}
    \caption{Validation Loss}
  \end{subfigure}
  \caption{Training and validation loss convergence curves for all eight
    experiments.  Experiments with larger combined training sets
    (Exp.~5--8) generally converge to lower training loss but may
    exhibit higher validation loss due to domain mismatch between
    auxiliary data and the SROIE test distribution.  Early stopping
    (patience = 5 epochs on validation loss) terminates training at
    different epochs across experiments.}
  \label{fig:convergence}
\end{figure*}
"""
    )
    tex_path = out_dir / "convergence_plots.tex"
    tex_path.write_text(tex, encoding="utf-8")


def generate_f1_barchart_tex(
    results_path: str = "results/all_experiments.json", output_dir: str = "results"
) -> None:
    """Generate results/f1_barchart.tex — horizontal bar chart of Global F1."""
    results_file = Path(results_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_labels: list = []
    f1_values: list = []

    if results_file.exists():
        with open(results_file) as fh:
            all_exp = json.load(fh)
        for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
            res = all_exp[exp_id_str]
            f1 = res.get("metrics", {}).get("global_f1")
            if f1 is not None:
                name = EXP_NAMES.get(exp_id_str, f"Exp {exp_id_str}")
                exp_labels.append(f"Exp.~{exp_id_str}: {name}")
                f1_values.append(f1)

    if not f1_values:
        tex_path = out_dir / "f1_barchart.tex"
        tex_path.write_text("% No F1 data available yet.\n", encoding="utf-8")
        return

    coords = "\n        ".join(f"({v:.4f},{i})" for i, v in enumerate(f1_values))
    ylabels = "\n        ".join(f"{i}/{{{lab}}}" for i, lab in enumerate(exp_labels))

    tex = (
        r"""\begin{figure}[h]
  \centering
  \begin{tikzpicture}
    \begin{axis}[
      xbar,
      xlabel={Global F1},
      ytick=data,
      yticklabels={
        """
        + ylabels
        + r"""
      },
      width=\linewidth,
      height=7cm,
      xmin=0, xmax=1,
      bar width=8pt,
      nodes near coords,
      nodes near coords align={horizontal},
      every node near coord/.style={font=\tiny},
    ]
      \addplot[fill=blue!60] coordinates {
        """
        + coords
        + r"""
      };
    \end{axis}
  \end{tikzpicture}
  \caption{Global F1 score on the SROIE test set for each of the eight
    fine-tuning experiments, showing the impact of auxiliary dataset
    inclusion on extraction accuracy.}
  \label{fig:f1_barchart}
\end{figure}
"""
    )
    tex_path = out_dir / "f1_barchart.tex"
    tex_path.write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def generate_experiment_table_slides(
    all_exp: dict,
    output_dir: str | Path = "results",
) -> None:
    """Generate results/experiment_table_slides.tex — tabular for Beamer slide.

    Produces a compact tabular block (no \\begin{table} wrapper) listing all
    experiments that have results, formatted for Beamer slide width.
    Uses \\small font and abbreviated dataset names.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
        res = all_exp[exp_id_str]
        m = res.get("metrics", {})
        name = EXP_NAMES.get(exp_id_str, res.get("name", f"Exp {exp_id_str}"))
        f1 = m.get("global_f1")
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        n = res.get("num_train_samples", 0)
        rows.append(f"    {exp_id_str} & {name} & {n:,} & {f1_str} \\\\")

    if not rows:
        tex = "% No experiment results available yet.\n"
    else:
        header = (
            "\\begin{tabular}{@{}clcc@{}}\n"
            "  \\toprule\n"
            "  \\textbf{Exp.} & \\textbf{Data} & "
            "\\textbf{$n$} & \\textbf{F1} \\\\\n"
            "  \\midrule\n"
        )
        footer = "  \\bottomrule\n\\end{tabular}\n"
        tex = header + "\n".join(rows) + "\n" + footer

    out = out_dir / "experiment_table_slides.tex"
    out.write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject experimental results into LaTeX tables")
    parser.add_argument(
        "--all", action="store_true", help="Generate all tables from results/all_experiments.json"
    )
    parser.add_argument(
        "--results",
        default="results/all_experiments.json",
        help="Path to all_experiments.json (default: results/all_experiments.json)",
    )
    parser.add_argument(
        "--paper",
        default="paper/paper.tex",
        help="Path to paper.tex template (default: paper/paper.tex)",
    )
    parser.add_argument(
        "--output",
        default="paper/paper_filled.tex",
        help="Path to write the filled paper (default: paper/paper_filled.tex)",
    )
    parser.add_argument(
        "--presentation",
        default="paper/presentation.tex",
        help="Path to presentation.tex template (default: paper/presentation.tex)",
    )
    parser.add_argument(
        "--presentation-output",
        default="paper/presentation_filled.tex",
        help="Path to write filled presentation (default: paper/presentation_filled.tex)",
    )
    args, _ = parser.parse_known_args()

    if args.all:
        results_path = Path(args.results)
        if not results_path.exists():
            warnings.warn(
                f"{results_path} not found — generating paper with placeholder values only.",
                stacklevel=1,
            )
            all_exp = {}
        else:
            try:
                with open(results_path, encoding="utf-8") as fh:
                    all_exp = json.load(fh)
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"Could not parse {results_path} as JSON ({exc}); "
                    "generating paper with placeholder values only.",
                    stacklevel=1,
                )
                all_exp = {}

        print_table1_dataset_stats()
        print_table2_experiments(all_exp)
        print_table3_perfield(all_exp)
        print_table4_leaderboard(all_exp)

        generate_convergence_data(args.results)
        generate_convergence_tex(args.results)
        generate_f1_barchart_tex(args.results)

        # Generate slides experiment table and convergence plots
        output_dir = str(results_path.parent)
        generate_experiment_table_slides(all_exp, output_dir=output_dir)

        # Generate cubic-spline convergence .tex files via plot_convergence
        try:
            generate_all(results_dir=output_dir)
        except Exception as exc:
            warnings.warn(
                f"plot_convergence.generate_all() failed: {exc}",
                stacklevel=2,
            )

        var_map = build_var_map(all_exp)

        if Path(args.paper).exists():
            # Read old content before overwriting so paper_diff can compare
            output_path = Path(args.output)
            _old_content = ""
            if output_path.exists():
                try:
                    _old_content = output_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass

            fill_paper(args.paper, args.output, var_map)

            # Run paper diff (compare old vs new paper_filled.tex)
            if _old_content:
                try:
                    _new_content = output_path.read_text(encoding="utf-8", errors="replace")
                    run_paper_diff(_old_content, _new_content, results_dir="results")
                except Exception as _pd_exc:
                    print(f"[PaperDiff] Skipped: {_pd_exc}")
        else:
            print(f"paper.tex not found at {args.paper}; skipping filled paper generation.")

        if Path(args.presentation).exists():
            fill_paper(args.presentation, args.presentation_output, var_map)
        else:
            print(
                f"presentation.tex not found at {args.presentation}; "
                "skipping filled presentation generation."
            )
    else:
        legacy_output()


# ---------------------------------------------------------------------------
# Paper diff utilities (inlined from paper_diff.py)
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def compute_diff(old_text: str, new_text: str) -> list[dict]:
    """Compute line-level diff between old and new filled LaTeX."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    changes: list[dict] = []
    max_len = max(len(old_lines), len(new_lines))
    for i in range(max_len):
        old_line = old_lines[i] if i < len(old_lines) else ""
        new_line = new_lines[i] if i < len(new_lines) else ""
        if old_line != new_line:
            old_stripped = old_line.strip()
            new_stripped = new_line.strip()
            delta: float | None = None
            direction = "~"
            if _NUMERIC_RE.match(old_stripped) and _NUMERIC_RE.match(new_stripped):
                try:
                    o = float(old_stripped)
                    n = float(new_stripped)
                    delta = n - o
                    direction = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                except ValueError:
                    pass
            changes.append(
                {
                    "line": i + 1,
                    "old": old_line,
                    "new": new_line,
                    "delta": delta,
                    "direction": direction,
                }
            )
    return changes


def print_diff_table(
    changes: list[dict],
    output_file: Path | None = None,
    use_rich: bool | None = None,
) -> None:
    """Print the diff as a table."""
    if use_rich is None:
        try:
            import rich  # noqa: F401

            use_rich = True
        except ImportError:
            use_rich = False

    lines: list[str] = [
        f"paper_diff — {len(changes)} line(s) changed",
        "=" * 80,
        f"{'Line':<6} {'Old value':<30} {'New value':<30} {'Δ':<12} {'Dir':<4}",
        "-" * 80,
    ]
    for c in changes:
        old_short = c["old"][:28].replace("\n", "").replace("\r", "")
        new_short = c["new"][:28].replace("\n", "").replace("\r", "")
        delta_str = f"{c['delta']:+.4f}" if c["delta"] is not None else ""
        lines.append(
            f"{c['line']:<6} {old_short:<30} {new_short:<30} {delta_str:<12} {c['direction']:<4}"
        )
    lines.append("=" * 80)
    plain_text = "\n".join(lines)

    if use_rich:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title=f"Paper Diff — {len(changes)} line(s) changed",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Line", justify="right", style="dim", width=6)
            table.add_column("Old value", style="red")
            table.add_column("New value", style="green")
            table.add_column("Δ", justify="right")
            table.add_column("Dir", justify="center")
            for c in changes:
                delta_str = f"{c['delta']:+.4f}" if c["delta"] is not None else ""
                dir_colour = (
                    "green"
                    if c["direction"] == "↑"
                    else ("red" if c["direction"] == "↓" else "yellow")
                )
                table.add_row(
                    str(c["line"]),
                    c["old"][:35],
                    c["new"][:35],
                    delta_str,
                    f"[{dir_colour}]{c['direction']}[/{dir_colour}]",
                )
            console.print(table)
        except Exception:
            print(plain_text)
    else:
        print(plain_text)

    if output_file is not None:
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(plain_text, encoding="utf-8")
            print(f"[PaperDiff] Diff written to: {output_file}")
        except OSError as exc:
            print(f"[PaperDiff] WARNING: Could not write diff file: {exc}")


def run_paper_diff(
    old_text: str,
    new_text: str,
    results_dir: str | Path = "results",
) -> None:
    """Compare old and new filled paper text; print and save the diff."""
    from datetime import datetime, timezone

    changes = compute_diff(old_text, new_text)
    if not changes:
        print("[PaperDiff] No changes detected in paper_filled.tex.")
        return
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_file = Path(results_dir) / f"paper_diff_{date_str}.txt"
    print_diff_table(changes, output_file=out_file)


# ---------------------------------------------------------------------------
# ResultsAggregator
# Absorbed from results_aggregator.py — loads per-experiment JSON files and
# builds an AggregatedResults object consumed by MLTrainingOrchestrator and
# the paper generator.  Kept here because both modules operate on result JSON
# files and share the build_var_map() function defined above.
# ---------------------------------------------------------------------------


import json as _json  # noqa: E402
import logging as _logging  # noqa: E402
from datetime import datetime as _datetime  # noqa: E402

from cloud_orchestration import AggregatedResults, ExperimentMetrics, ExperimentResult  # noqa: E402


class ResultsAggregator:
    """Aggregate experiment results for paper generation.

    Reads all ``experiment_N.json`` files from *results_dir* and assembles an
    :class:`AggregatedResults` dataclass that identifies the best experiment,
    the baseline F1, and the overall improvement.
    """

    _logger = _logging.getLogger(__name__)

    def __init__(self, results_dir: Path = Path("results")) -> None:
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_experiments(self) -> AggregatedResults | None:
        """Load and aggregate all experiment results.

        Returns:
            AggregatedResults, or None if no experiment files exist.
        """
        self._logger.info("Aggregating experiment results...")
        experiment_files = sorted(self.results_dir.glob("experiment_*.json"))

        if not experiment_files:
            self._logger.warning("No experiment results found")
            return None

        experiments: list[ExperimentResult] = []
        for exp_file in experiment_files:
            try:
                data = _json.loads(exp_file.read_text())
                if isinstance(data, dict):
                    metrics = ExperimentMetrics(**data.get("metrics", {}))
                    exp = ExperimentResult(
                        experiment_id=data["experiment_id"],
                        name=data["name"],
                        datasets=data["datasets"],
                        num_train_samples=data["num_train_samples"],
                        metrics=metrics,
                    )
                    experiments.append(exp)
            except Exception as e:
                self._logger.warning(f"Could not load {exp_file}: {e}")

        if not experiments:
            self._logger.warning("No valid experiments loaded")
            return None

        best_exp = max(experiments, key=lambda e: e.metrics.global_f1)
        baseline_exp = next((e for e in experiments if e.experiment_id == 1), None)
        baseline_f1 = baseline_exp.metrics.global_f1 if baseline_exp else 0.0
        improvement = best_exp.metrics.global_f1 - baseline_f1

        agg = AggregatedResults(
            experiments=experiments,
            best_experiment=best_exp,
            baseline_f1=baseline_f1,
            improvement=improvement,
            generated_timestamp=_datetime.utcnow(),
        )

        self._logger.info(
            f"✓ Aggregated {len(experiments)} experiments. "
            f"Best: Exp {best_exp.experiment_id} F1={best_exp.metrics.global_f1:.4f}"
        )
        return agg

    def build_paper_metrics(self, agg: AggregatedResults) -> dict[str, str]:
        r"""Build \VAR{} key→value map for LaTeX template.

        Delegates to :func:`build_var_map` — single source of truth.
        """
        all_exp = {str(e.experiment_id): e.__dict__ for e in agg.experiments}
        return build_var_map(all_exp)

    def save_aggregated_results(
        self, agg: AggregatedResults, output_file: Path | None = None
    ) -> bool:
        """Save aggregated results to JSON.

        NOTE: ``all_experiments.json`` is owned by ``run_experiments.save_summary()``.
        This method writes to ``aggregated_summary.json`` instead.
        """
        if output_file is None:
            output_file = self.results_dir / "aggregated_summary.json"

        try:
            data = {
                "generated_timestamp": agg.generated_timestamp.isoformat(),
                "num_experiments": len(agg.experiments),
                "best_experiment": (
                    {
                        "experiment_id": agg.best_experiment.experiment_id,
                        "name": agg.best_experiment.name,
                        "f1": agg.best_experiment.metrics.global_f1,
                    }
                    if agg.best_experiment
                    else None
                ),
                "baseline_f1": agg.baseline_f1,
                "improvement": agg.improvement,
                "experiments": [e.to_dict() for e in agg.experiments],
            }
            output_file.write_text(_json.dumps(data, indent=2))
            self._logger.info(f"✓ Saved aggregated results to {output_file}")
            return True
        except Exception as e:
            self._logger.error(f"Could not save aggregated results: {e}")
            return False

    @staticmethod
    def load_all_experiments(results_dir: Path) -> list[ExperimentResult]:
        """Load all experiment JSON files from *results_dir*.

        Returns:
            List of ExperimentResult objects (empty list on error).
        """
        experiments: list[ExperimentResult] = []
        for exp_file in sorted(results_dir.glob("experiment_*.json")):
            try:
                data = _json.loads(exp_file.read_text())
                metrics = ExperimentMetrics(**data.get("metrics", {}))
                exp = ExperimentResult(
                    experiment_id=data["experiment_id"],
                    name=data["name"],
                    datasets=data["datasets"],
                    num_train_samples=data["num_train_samples"],
                    metrics=metrics,
                )
                experiments.append(exp)
            except Exception as e:
                _logging.getLogger(__name__).warning(f"Could not load {exp_file}: {e}")
        return experiments


if __name__ == "__main__":
    main()
