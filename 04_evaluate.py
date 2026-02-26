"""
04_evaluate.py
==============
Evaluates both models on the test set and computes:
  - CER  (Character Error Rate)   ← raw OCR quality
  - WER  (Word Error Rate)        ← word-level accuracy
  - F1   (field-level extraction) ← key-value accuracy
  - Precision / Recall per field
  - Inference latency             ← ms per image
  - Results saved to results/metrics.json
"""

import json, time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from jiwer import cer, wer
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    TrOCRProcessor,
)
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────────────
DONUT_MODEL   = Path("models/donut_finetuned/best")
TROCR_MODEL   = Path("models/trocr_finetuned/best")
YOLO_MODEL    = Path("models/yolo_finetuned/run/weights/best.pt")
TEST_DONUT    = Path("data/donut/test")
TEST_TROCR    = Path("data/trocr/test")
RESULTS_DIR   = Path("results")
TASK_TOKEN    = "<s_receipt>"
TARGET_KEYS   = ["company", "date", "address", "total"]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH    = 512
NUM_BEAMS     = 4

RESULTS_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# DONUT Inference
# ════════════════════════════════════════════════════════════════════════════
def load_donut():
    processor = DonutProcessor.from_pretrained(DONUT_MODEL)
    model     = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL).to(DEVICE)
    model.eval()
    return processor, model


def predict_donut(processor, model, image: Image.Image) -> dict:
    """Returns dict of extracted key-values from a receipt image."""
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    decoder_input_ids = processor.tokenizer(
        TASK_TOKEN, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids  = decoder_input_ids,
            max_length         = MAX_LENGTH,
            num_beams          = NUM_BEAMS,
            early_stopping     = True,
        )

    seq = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(
          processor.tokenizer.pad_token, "").strip()

    # Parse XML-like tags  → dict
    result = {}
    for key in TARGET_KEYS:
        start_tag = f"<s_{key}>"
        end_tag   = f"</s_{key}>"
        if start_tag in seq and end_tag in seq:
            val = seq.split(start_tag)[1].split(end_tag)[0].strip()
            result[key] = val
        else:
            result[key] = ""
    return result


def evaluate_donut(processor, model):
    print("\n=== Evaluating DONUT ===")
    samples, latencies = [], []
    predictions, ground_truths = [], []

    meta_path = TEST_DONUT / "metadata.jsonl"
    with open(meta_path) as f:
        samples = [json.loads(l) for l in f]

    all_pred_kv, all_gt_kv = [], []

    for sample in tqdm(samples, desc="DONUT inference"):
        image  = Image.open(TEST_DONUT / sample["file_name"]).convert("RGB")
        gt_kv  = json.loads(sample["ground_truth"])["gt_parse"]

        t0   = time.perf_counter()
        pred = predict_donut(processor, model, image)
        lat  = (time.perf_counter() - t0) * 1000

        latencies.append(lat)
        all_pred_kv.append(pred)
        all_gt_kv.append(gt_kv)

        # For CER/WER: concatenate all field values
        pred_text = " ".join(pred.values())
        gt_text   = " ".join(gt_kv.get(k, "") for k in TARGET_KEYS)
        predictions.append(pred_text)
        ground_truths.append(gt_text)

    metrics = compute_metrics(predictions, ground_truths, all_pred_kv, all_gt_kv, latencies)
    print_metrics("DONUT", metrics)
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# TrOCR + YOLO Inference
# ════════════════════════════════════════════════════════════════════════════
def load_trocr_yolo():
    yolo      = YOLO(str(YOLO_MODEL))
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    trocr     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(DEVICE)
    trocr.eval()
    return yolo, processor, trocr


def predict_trocr_yolo(yolo_model, processor, trocr_model, image: Image.Image) -> str:
    """
    Detects text regions with YOLO, transcribes each with TrOCR.
    Returns full concatenated text.
    """
    W, H = image.size

    # YOLO detection
    results = yolo_model.predict(image, conf=0.25, verbose=False)
    boxes   = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) else []

    if len(boxes) == 0:
        return ""

    # Sort boxes top-to-bottom, left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    all_text = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        crop = image.crop((x1, y1, x2, y2))
        pixel_values = processor(crop, return_tensors="pt").pixel_values.to(DEVICE)

        with torch.no_grad():
            generated = trocr_model.generate(
                pixel_values,
                max_length = 128,
                num_beams  = 4,
            )

        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        all_text.append(text)

    return "\n".join(all_text)


def evaluate_trocr_yolo(yolo_model, processor, trocr_model):
    """
    For TrOCR+YOLO we measure CER/WER on full OCR output.
    Field-level F1 uses simple substring matching (no structured output).
    """
    print("\n=== Evaluating TrOCR + YOLO ===")
    latencies = []
    predictions, ground_truths = [], []
    all_pred_kv, all_gt_kv = [], []

    # Load test samples (use trocr metadata which has full receipt text per image)
    # We aggregate line-level texts per image
    image_texts = {}
    with open(TEST_TROCR / "metadata.jsonl") as f:
        for line in f:
            s = json.loads(line)
            img_key = s["file_name"].rsplit("_", 1)[0]  # strip line index
            image_texts.setdefault(img_key, []).append(s["text"])

    # Load DONUT test for GT key-values (same test images)
    donut_gt = {}
    with open(Path("data/donut/test") / "metadata.jsonl") as f:
        for line in f:
            s   = json.loads(line)
            key = Path(s["file_name"]).stem
            donut_gt[key] = json.loads(s["ground_truth"])["gt_parse"]

    # Load test images
    test_images = sorted(Path("data/yolo/images/test").glob("*.png"))

    for img_path in tqdm(test_images, desc="TrOCR+YOLO inference"):
        image = Image.open(img_path).convert("RGB")
        stem  = img_path.stem

        t0   = time.perf_counter()
        pred_text = predict_trocr_yolo(yolo_model, processor, trocr_model, image)
        lat  = (time.perf_counter() - t0) * 1000

        latencies.append(lat)

        # GT full text  (join GT lines)
        gt_lines = image_texts.get(stem, [])
        gt_text  = "\n".join(gt_lines)

        predictions.append(pred_text)
        ground_truths.append(gt_text)

        # Field-level: use substring heuristics
        gt_kv = donut_gt.get(stem, {k: "" for k in TARGET_KEYS})
        pred_kv = extract_fields_from_text(pred_text, gt_kv)
        all_pred_kv.append(pred_kv)
        all_gt_kv.append(gt_kv)

    metrics = compute_metrics(predictions, ground_truths, all_pred_kv, all_gt_kv, latencies)
    print_metrics("TrOCR+YOLO", metrics)
    return metrics


def extract_fields_from_text(text: str, gt_kv: dict) -> dict:
    """Simple heuristic: check if GT value appears in predicted text."""
    result = {}
    lines  = text.lower().split("\n")
    for key, gt_val in gt_kv.items():
        if not gt_val:
            result[key] = ""
            continue
        # Return GT value if found, else empty
        found = any(gt_val.lower() in line for line in lines)
        result[key] = gt_val if found else ""
    return result


# ════════════════════════════════════════════════════════════════════════════
# Shared Metrics
# ════════════════════════════════════════════════════════════════════════════
def compute_metrics(predictions, ground_truths, pred_kvs, gt_kvs, latencies):
    # Filter empty GT
    pairs = [(p, g, pk, gk) for p, g, pk, gk in
             zip(predictions, ground_truths, pred_kvs, gt_kvs) if g.strip()]
    preds, gts, pred_kv_list, gt_kv_list = zip(*pairs) if pairs else ([], [], [], [])

    metrics = {}

    # CER / WER
    if preds:
        metrics["cer"] = cer(list(gts), list(preds))
        metrics["wer"] = wer(list(gts), list(preds))
    else:
        metrics["cer"] = metrics["wer"] = 1.0

    # Field-level F1 per key
    field_metrics = {}
    for key in TARGET_KEYS:
        tp = fp = fn = 0
        for pred_kv, gt_kv in zip(pred_kv_list, gt_kv_list):
            pred_val = pred_kv.get(key, "").strip().lower()
            gt_val   = gt_kv.get(key, "").strip().lower()
            if gt_val == "" and pred_val == "":
                continue
            if pred_val == gt_val and pred_val != "":
                tp += 1
            elif pred_val != "" and gt_val == "":
                fp += 1
            elif pred_val == "" and gt_val != "":
                fn += 1
            else:
                # both non-empty but different → FP + FN
                fp += 1
                fn += 1

        prec   = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1     = 2 * prec * recall / (prec + recall + 1e-9)
        field_metrics[key] = {"precision": prec, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    metrics["field_metrics"]  = field_metrics
    metrics["macro_f1"]       = np.mean([v["f1"] for v in field_metrics.values()])
    metrics["latency_mean_ms"]= np.mean(latencies)
    metrics["latency_p95_ms"] = np.percentile(latencies, 95)
    metrics["n_samples"]      = len(preds)

    return metrics


def print_metrics(name, metrics):
    print(f"\n{'─'*50}")
    print(f" {name}")
    print(f"{'─'*50}")
    print(f"  CER:            {metrics['cer']:.4f}")
    print(f"  WER:            {metrics['wer']:.4f}")
    print(f"  Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"  Latency (mean): {metrics['latency_mean_ms']:.1f} ms")
    print(f"  Latency  (p95): {metrics['latency_p95_ms']:.1f} ms")
    print(f"\n  Field-level breakdown:")
    for key, fm in metrics["field_metrics"].items():
        print(f"    {key:10s}  P={fm['precision']:.3f}  R={fm['recall']:.3f}  F1={fm['f1']:.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}

    # DONUT
    processor_d, model_d = load_donut()
    results["donut"] = evaluate_donut(processor_d, model_d)
    del processor_d, model_d
    torch.cuda.empty_cache()

    # TrOCR + YOLO
    yolo_m, processor_t, model_t = load_trocr_yolo()
    results["trocr_yolo"] = evaluate_trocr_yolo(yolo_m, processor_t, model_t)
    del yolo_m, processor_t, model_t
    torch.cuda.empty_cache()

    # Save
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics saved → {out_path}")
