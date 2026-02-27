"""
04_evaluate.py — Unified evaluation for DONUT and TrOCR+YOLO architectures.

FIX: Previous version used separate metric conventions (CER/WER for TrOCR
vs structured F1 for DONUT).  This version uses the SAME metrics for both
architectures — the official SROIE Task-3 entity-level F1, precision,
recall, NED, and exact match — enabling fair cross-architecture comparison.

FIX: Removed early_stopping=True from DONUT generate() calls — invalid
with num_beams=1 (greedy decoding) and produces deprecation warnings.

FIX: Added GPU cleanup between model evaluations to prevent OOM.

FIX: Both architectures are evaluated on the SAME 63 SROIE test images.

FIX: Imports constants from shared module instead of duplicating.
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from constants import FIELDS, IMAGE_EXTS, MAX_LENGTH, SEED

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SROIE_DATA_DIR = Path(os.environ.get(
    "SROIE_DATA_DIR", "/workspace/ICDAR-2019-SROIE/data"
))


# ════════════════════════════════════════════════════════════════════════════
# Shared metric computation (SROIE Task-3 compatible)
# ════════════════════════════════════════════════════════════════════════════
def compute_sroie_metrics(
    predictions: List[Dict[str, str]],
    ground_truths: List[Dict[str, str]],
) -> Dict[str, float]:
    """Compute SROIE Task-3 metrics: global F1, per-field F1, NED, exact match.

    Uses the same computation as evaluate.py:compute_metrics() for consistency
    across DONUT and TrOCR+YOLO results.
    """
    try:
        import editdistance
    except ImportError:
        editdistance = None

    tp, total_pred, total_gt = 0, 0, 0
    per_field = {f: {"tp": 0, "pred": 0, "gt": 0, "ned": []} for f in FIELDS}
    exact_match_all = []

    for pred, gt in zip(predictions, ground_truths):
        all_correct = True
        for f in FIELDS:
            p_val = str(pred.get(f, "")).strip().lower()
            g_val = str(gt.get(f, "")).strip().lower()

            if g_val:
                total_gt += 1
                per_field[f]["gt"] += 1
            if p_val:
                total_pred += 1
                per_field[f]["pred"] += 1
            if p_val and g_val and p_val == g_val:
                tp += 1
                per_field[f]["tp"] += 1
            elif not p_val and not g_val:
                pass
            else:
                all_correct = False

            # NED computation
            if editdistance is not None:
                if len(g_val) == 0:
                    ned = 0.0 if len(p_val) == 0 else 1.0
                else:
                    ned = editdistance.eval(p_val, g_val) / max(len(p_val), len(g_val))
                per_field[f]["ned"].append(ned)

        exact_match_all.append(int(all_correct))

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    summary = {
        "global_precision": round(precision, 4),
        "global_recall": round(recall, 4),
        "global_f1": round(f1, 4),
        "overall_exact_match": round(float(np.mean(exact_match_all)), 4) if exact_match_all else 0.0,
    }

    for f in FIELDS:
        fp = per_field[f]
        p = fp["tp"] / fp["pred"] if fp["pred"] > 0 else 0.0
        r = fp["tp"] / fp["gt"] if fp["gt"] > 0 else 0.0
        f_score = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        summary[f"{f}_f1"] = round(f_score, 4)
        summary[f"{f}_ned"] = round(float(np.mean(fp["ned"])), 4) if fp["ned"] else 1.0

    return summary


# ════════════════════════════════════════════════════════════════════════════
# Load SROIE test set (shared by both architectures)
# ════════════════════════════════════════════════════════════════════════════
def load_test_samples() -> List[Tuple[Path, Dict[str, str]]]:
    """Load the 63 SROIE test images + ground truth.

    Returns list of (image_path, gt_dict) tuples.  Both DONUT and TrOCR+YOLO
    are evaluated on this EXACT same set for fair comparison.
    """
    test_img_dir = SROIE_DATA_DIR / "test_img"
    test_key_dir = SROIE_DATA_DIR / "test_key"

    samples = []
    if not test_img_dir.exists():
        print(f"  WARNING: test_img/ not found at {test_img_dir}")
        return samples

    for img_path in sorted(test_img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        gt = {}
        key_txt = test_key_dir / f"{img_path.stem}.txt"
        if key_txt.exists():
            lines = key_txt.read_text(encoding="utf-8").strip().splitlines()
            if len(lines) >= 4:
                gt = {
                    "company": lines[0].strip(),
                    "date":    lines[1].strip(),
                    "address": lines[2].strip(),
                    "total":   lines[3].strip(),
                }
        key_json = test_key_dir / f"{img_path.stem}.json"
        if not gt and key_json.exists():
            try:
                gt = json.loads(key_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        if gt:
            samples.append((img_path, gt))

    return samples


# ════════════════════════════════════════════════════════════════════════════
# Evaluate DONUT on test set
# ════════════════════════════════════════════════════════════════════════════
def evaluate_donut_on_test(
    model_path: str,
    test_samples: List[Tuple[Path, Dict[str, str]]],
) -> Dict:
    """Evaluate a DONUT model on the SROIE test set. Returns metrics dict."""
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from evaluate import load_model_with_tied_weights, _unwrap_prediction

    processor = DonutProcessor.from_pretrained(model_path)
    model = load_model_with_tied_weights(model_path, device=DEVICE)

    predictions = []
    ground_truths = [s[1] for s in test_samples]
    latencies = []
    parse_failures = 0

    with torch.no_grad():
        for img_path, gt in tqdm(test_samples, desc="DONUT eval"):
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
            decoder_input_ids = processor.tokenizer(
                "<s_sroie>", add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(DEVICE)

            t0 = time.perf_counter()
            # FIX: No early_stopping=True — invalid with num_beams=1
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=MAX_LENGTH,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            sequence = processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(processor.tokenizer.eos_token, "")
            sequence = sequence.replace(processor.tokenizer.pad_token, "").strip()

            try:
                parsed = processor.token2json(sequence)
                parsed = _unwrap_prediction(parsed, "<s_sroie>")
            except Exception:
                parsed = {}
                parse_failures += 1

            predictions.append(parsed)

    metrics = compute_sroie_metrics(predictions, ground_truths)
    metrics["parse_failures"] = parse_failures
    metrics["num_samples"] = len(test_samples)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 1) if latencies else 0.0

    # GPU cleanup
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Evaluate TrOCR+YOLO on test set
# ════════════════════════════════════════════════════════════════════════════
def evaluate_trocr_yolo_on_test(
    yolo_weights: str,
    trocr_model_path: str,
    test_samples: List[Tuple[Path, Dict[str, str]]],
) -> Dict:
    """Evaluate TrOCR+YOLO pipeline on the SROIE test set. Returns metrics dict."""
    from ultralytics import YOLO
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from importlib import import_module

    # Import the inference function from 03_train_trocr_yolo.py
    trocr_yolo_module = import_module("03_train_trocr_yolo")
    run_pipeline = trocr_yolo_module.run_trocr_yolo_inference

    yolo_model = YOLO(str(yolo_weights))
    trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_path)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_path).to(DEVICE)
    trocr_model.eval()

    predictions = []
    ground_truths = [s[1] for s in test_samples]
    latencies = []

    with torch.no_grad():
        for img_path, gt in tqdm(test_samples, desc="TrOCR+YOLO eval"):
            t0 = time.perf_counter()
            pred = run_pipeline(img_path, yolo_model, trocr_model, trocr_processor)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            predictions.append(pred)

    metrics = compute_sroie_metrics(predictions, ground_truths)
    metrics["num_samples"] = len(test_samples)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 1) if latencies else 0.0

    # GPU cleanup
    del yolo_model, trocr_model, trocr_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ── Print metrics ────────────────────────────────────────────────────────────
def print_metrics(name: str, metrics: Dict) -> None:
    """Pretty-print evaluation metrics in structured format."""
    print(f"\n  {'='*55}")
    print(f"  {name} Results")
    print(f"  {'='*55}")
    print(f"  Global F1:        {metrics.get('global_f1', 0):.4f}")
    print(f"  Global Precision: {metrics.get('global_precision', 0):.4f}")
    print(f"  Global Recall:    {metrics.get('global_recall', 0):.4f}")
    print(f"  Exact Match:      {metrics.get('overall_exact_match', 0):.4f}")
    print(f"  Num Samples:      {metrics.get('num_samples', 0)}")
    if "mean_latency_ms" in metrics:
        print(f"  Mean Latency:     {metrics['mean_latency_ms']:.1f} ms/image")
    print(f"  {'-'*55}")
    for f in FIELDS:
        f1 = metrics.get(f"{f}_f1", 0)
        ned = metrics.get(f"{f}_ned", 1)
        print(f"  {f:12s}  F1={f1:.4f}  NED={ned:.4f}")
    print(f"  {'='*55}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_samples = load_test_samples()
    print(f"Loaded {len(test_samples)} test samples")

    results = {}

    # Evaluate DONUT (if model exists)
    donut_model = Path("models/donut_finetuned/best")
    if donut_model.exists():
        results["donut"] = evaluate_donut_on_test(str(donut_model), test_samples)
        print_metrics("DONUT", results["donut"])
    else:
        print(f"  DONUT model not found at {donut_model} — skipping")

    # Evaluate TrOCR+YOLO (if models exist)
    yolo_weights = Path("models/yolo_finetuned/run/weights/best.pt")
    trocr_model = Path("models/trocr_finetuned/best")
    if yolo_weights.exists() and trocr_model.exists():
        results["trocr_yolo"] = evaluate_trocr_yolo_on_test(
            str(yolo_weights), str(trocr_model), test_samples
        )
        print_metrics("TrOCR+YOLO", results["trocr_yolo"])
    else:
        print(f"  TrOCR+YOLO models not found — skipping")

    # Save combined results
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved -> {out_path}")
