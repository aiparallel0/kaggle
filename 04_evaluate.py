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

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from constants import DEVICE, FIELDS, IMAGE_EXTS, MAX_LENGTH, _gpu_cleanup
from dataset_loaders import _load_key_file

# ── Config ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")

SROIE_DATA_DIR = Path(os.environ.get("SROIE_DATA_DIR", "/workspace/ICDAR-2019-SROIE/data"))


# ════════════════════════════════════════════════════════════════════════════
# Shared metric computation (SROIE Task-3 compatible)
# ════════════════════════════════════════════════════════════════════════════
# Import compute_metrics from evaluate.py — single source of truth for SROIE
# Task-3 F1/NED/exact-match computation shared across DONUT and TrOCR+YOLO.
from donut_evaluator import compute_metrics as compute_sroie_metrics  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Load SROIE test set (shared by both architectures)
# ════════════════════════════════════════════════════════════════════════════
def load_test_samples() -> list[tuple[Path, dict[str, str]]]:
    """Load the 63 SROIE test images + ground truth.

    Returns list of (image_path, gt_dict) tuples.  Both DONUT and TrOCR+YOLO
    are evaluated on this EXACT same set for fair comparison.

    Uses _load_key_file from dataset_loaders (single source of truth for
    SROIE key file parsing) instead of duplicating the .txt/.json logic.
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

        gt = _load_key_file(test_key_dir, img_path.stem)
        if gt:
            samples.append((img_path, gt))

    return samples


# ════════════════════════════════════════════════════════════════════════════
# Evaluate DONUT on test set
# ════════════════════════════════════════════════════════════════════════════
def evaluate_donut_on_test(
    model_path: str,
    test_samples: list[tuple[Path, dict[str, str]]],
) -> dict:
    """Evaluate a DONUT model on the SROIE test set. Returns metrics dict."""
    from transformers import DonutProcessor

    from donut_evaluator import _parse_sroie_output, load_model_with_tied_weights

    processor = DonutProcessor.from_pretrained(model_path)
    model = load_model_with_tied_weights(model_path, device=DEVICE)

    predictions = []
    ground_truths = [s[1] for s in test_samples]
    latencies = []
    parse_failures = 0

    with torch.no_grad():
        for img_path, _gt in tqdm(test_samples, desc="DONUT eval"):
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
                parsed = _parse_sroie_output(sequence)
            except Exception:
                parsed = {}
                parse_failures += 1

            predictions.append(parsed)

    metrics = compute_sroie_metrics(predictions, ground_truths)
    metrics["parse_failures"] = parse_failures
    metrics["num_samples"] = len(test_samples)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 1) if latencies else 0.0

    # GPU cleanup
    _gpu_cleanup(model, processor)

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Evaluate TrOCR+YOLO on test set
# ════════════════════════════════════════════════════════════════════════════
def evaluate_trocr_yolo_on_test(
    yolo_weights: str,
    trocr_model_path: str,
    test_samples: list[tuple[Path, dict[str, str]]],
) -> dict:
    """Evaluate TrOCR+YOLO pipeline on the SROIE test set. Returns metrics dict."""
    from importlib import import_module

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from ultralytics import YOLO

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
        for img_path, _gt in tqdm(test_samples, desc="TrOCR+YOLO eval"):
            t0 = time.perf_counter()
            pred = run_pipeline(img_path, yolo_model, trocr_model, trocr_processor)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            predictions.append(pred)

    metrics = compute_sroie_metrics(predictions, ground_truths)
    metrics["num_samples"] = len(test_samples)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 1) if latencies else 0.0

    # GPU cleanup
    _gpu_cleanup(yolo_model, trocr_model, trocr_processor)

    return metrics


# ── Print metrics ────────────────────────────────────────────────────────────
def print_metrics(name: str, metrics: dict) -> None:
    """Pretty-print evaluation metrics in structured format."""
    print(f"\n  {'=' * 55}")
    print(f"  {name} Results")
    print(f"  {'=' * 55}")
    print(f"  Global F1:        {metrics.get('global_f1', 0):.4f}")
    print(f"  Global Precision: {metrics.get('global_precision', 0):.4f}")
    print(f"  Global Recall:    {metrics.get('global_recall', 0):.4f}")
    print(f"  Exact Match:      {metrics.get('overall_exact_match', 0):.4f}")
    print(f"  Num Samples:      {metrics.get('num_samples', 0)}")
    if "mean_latency_ms" in metrics:
        print(f"  Mean Latency:     {metrics['mean_latency_ms']:.1f} ms/image")
    print(f"  {'-' * 55}")
    for f in FIELDS:
        f1 = metrics.get(f"{f}_f1", 0)
        ned = metrics.get(f"{f}_ned", 1)
        print(f"  {f:12s}  F1={f1:.4f}  NED={ned:.4f}")
    print(f"  {'=' * 55}")


def generate_comparison_report(results: dict) -> None:
    """Generate a detailed HTML comparison report of all evaluated models."""
    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Model Evaluation Report</title>",
        "<style>",
        "  body { font-family: monospace; margin: 20px; }",
        "  table { border-collapse: collapse; margin: 20px 0; }",
        "  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "  th { background-color: #4CAF50; color: white; }",
        "  tr:nth-child(even) { background-color: #f2f2f2; }",
        "  .metric-high { color: green; font-weight: bold; }",
        "  .metric-low { color: red; }",
        "  h1, h2 { color: #333; }",
        "</style></head><body>",
        "<h1>Model Evaluation Report</h1>",
    ]

    if not results:
        html_lines.append("<p>No results to display.</p>")
    else:
        # Extract architectures and create comparison table
        html_lines.append("<h2>Global Performance Comparison</h2>")
        html_lines.append("<table><tr><th>Architecture</th><th>Global F1</th><th>Precision</th><th>Recall</th><th>Exact Match</th><th>Latency (ms)</th></tr>")

        for arch, metrics in results.items():
            f1 = metrics.get("global_f1", 0)
            prec = metrics.get("global_precision", 0)
            rec = metrics.get("global_recall", 0)
            em = metrics.get("overall_exact_match", 0)
            lat = metrics.get("mean_latency_ms", 0)

            f1_class = "metric-high" if f1 > 0.85 else "metric-low" if f1 < 0.7 else ""

            html_lines.append(
                f"<tr><td><strong>{arch}</strong></td><td class='{f1_class}'>{f1:.4f}</td>"
                f"<td>{prec:.4f}</td><td>{rec:.4f}</td><td>{em:.4f}</td><td>{lat:.1f}</td></tr>"
            )

        html_lines.append("</table>")

        # Per-field comparison
        html_lines.append("<h2>Per-Field Metrics</h2>")
        for arch, metrics in results.items():
            html_lines.append(f"<h3>{arch.upper()}</h3>")
            html_lines.append("<table><tr><th>Field</th><th>F1</th><th>NED</th></tr>")

            for field in FIELDS:
                f1 = metrics.get(f"{field}_f1", 0)
                ned = metrics.get(f"{field}_ned", 1)
                html_lines.append(f"<tr><td>{field}</td><td>{f1:.4f}</td><td>{ned:.4f}</td></tr>")

            html_lines.append("</table>")

    html_lines.extend(["</body></html>"])

    report_path = RESULTS_DIR / "evaluation_report.html"
    with open(report_path, "w") as f:
        f.write("\n".join(html_lines))

    print(f"\n📊 Detailed report saved -> {report_path}")


def generate_json_summary(results: dict) -> None:
    """Export evaluation results in structured JSON format."""
    summary = {
        "evaluation_timestamp": __import__("datetime").datetime.now().isoformat(),
        "num_test_samples": results.get("donut", {}).get("num_samples", 0),
        "architectures_evaluated": list(results.keys()),
        "results": results,
    }

    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📋 Summary saved -> {summary_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DONUT and TrOCR+YOLO models")
    parser.add_argument(
        "--donut-only",
        action="store_true",
        help="Only evaluate DONUT (skip TrOCR+YOLO)"
    )
    parser.add_argument(
        "--trocr-only",
        action="store_true",
        help="Only evaluate TrOCR+YOLO (skip DONUT)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML comparison report"
    )

    args = parser.parse_args()

    test_samples = load_test_samples()
    print(f"Loaded {len(test_samples)} test samples")

    results = {}

    # Evaluate DONUT (if model exists and not skipped)
    if not args.trocr_only:
        donut_model = Path("models/donut_finetuned/best")
        if donut_model.exists():
            results["donut"] = evaluate_donut_on_test(str(donut_model), test_samples)
            print_metrics("DONUT", results["donut"])
        else:
            print(f"  DONUT model not found at {donut_model} — skipping")

    # Evaluate TrOCR+YOLO (if models exist and not skipped)
    if not args.donut_only:
        yolo_weights = Path("models/yolo_finetuned/run/weights/best.pt")
        trocr_model = Path("models/trocr_finetuned/best")
        if yolo_weights.exists() and trocr_model.exists():
            results["trocr_yolo"] = evaluate_trocr_yolo_on_test(
                str(yolo_weights), str(trocr_model), test_samples
            )
            print_metrics("TrOCR+YOLO", results["trocr_yolo"])
        else:
            print("  TrOCR+YOLO models not found — skipping")

    # Save combined results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Metrics saved -> {out_path}")

    # Optional: generate report
    if args.report or True:  # Always generate summary now
        generate_json_summary(results)
        if results:
            generate_comparison_report(results)
