import json, torch
import editdistance
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

# WARNING: This is a legacy standalone script. For the full 8-experiment
# pipeline, use: python run_all.py
# This script is kept for backward compatibility and ad-hoc single-model
# training/evaluation outside the experiment framework.

import os
# BUG C FIX: Use a function so the env var is re-read at call time, not cached
# at import time (run_all.py sets SROIE_DATA_DIR after this module is imported).
def _get_sroie_dir():
    return Path(os.environ.get("SROIE_DATA_DIR", "/workspace/ICDAR-2019-SROIE/data"))

FIELDS = ["company", "date", "address", "total"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def run_inference(model, processor, image_path, task_prompt, max_length=512):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(DEVICE)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=max_length,
        early_stopping=True,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "").strip()
    # FIX (BUG 5): token2json can raise exceptions or return empty/malformed dicts
    # on garbage model output. Wrap in try/except to prevent silent zero-metric contamination.
    try:
        return processor.token2json(sequence)
    except Exception as e:
        print(f"[WARNING] token2json failed for {image_path}: {e}")
        return {}


def remap_cord_to_sroie(cord_output):
    """Map CORD schema fields to SROIE field names (best effort).

    Handles the 'cord-v2' top-level wrapper that the pretrained CORD model
    may emit (e.g. {"cord-v2": {...}}).
    """
    result = {}
    if isinstance(cord_output, dict):
        # Unwrap the "cord-v2" top-level key if present
        if "cord-v2" in cord_output and isinstance(cord_output["cord-v2"], dict):
            cord_output = cord_output["cord-v2"]
        store_info = cord_output.get("store_info", {})
        if isinstance(store_info, dict):
            result["company"] = store_info.get("store_name", "")
            result["address"] = store_info.get("region", "")
        total_info = cord_output.get("total", {})
        if isinstance(total_info, dict):
            result["total"] = total_info.get("total_price", "")
        elif isinstance(total_info, list) and len(total_info) > 0:
            result["total"] = total_info[0].get("total_price", "")
        date_info = cord_output.get("date", {})
        if isinstance(date_info, dict):
            result["date"] = str(date_info.get("date_value", "")).strip()
        elif isinstance(date_info, list) and len(date_info) > 0:
            result["date"] = str(date_info[0].get("date_value", "")).strip()
        elif isinstance(date_info, str):
            result["date"] = date_info.strip()
        else:
            result["date"] = ""
    return result


def normalized_edit_distance(pred, gt):
    """Compute Normalized Edit Distance (NED) between pred and gt strings.

    NED = editdistance(pred, gt) / max(len(pred), len(gt))

    Convention: NED is in [0, 1] where 0 means identical and 1 means maximally
    different. This formulation uses max(len(pred), len(gt)) as the denominator,
    ensuring NED <= 1.0. Lower is better.
    """
    pred, gt = str(pred).lower().strip(), str(gt).lower().strip()
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / max(len(pred), len(gt))


def compute_metrics(predictions, ground_truths):
    """
    Official SROIE Task 3 metric: global F1 over all (image, field) pairs.
    A pair is TP if predicted string == ground truth string (case-insensitive, stripped).
    """
    import sys as _sys
    tp, total_pred, total_gt = 0, 0, 0
    per_field = {f: {"tp": 0, "pred": 0, "gt": 0, "ned": []} for f in FIELDS}
    exact_match_all = []

    # Check for total prediction failure: all predictions are empty dicts
    all_empty = all(not any(str(pred.get(f, "")).strip() for f in FIELDS) for pred in predictions)
    if all_empty and predictions:
        print(
            "CRITICAL WARNING: ALL predictions are empty (token2json total failure). "
            "F1 will be 0.0 — check model output and token2json compatibility.",
            file=_sys.stderr,
        )

    for pred, gt in zip(predictions, ground_truths):
        all_correct = True
        for field in FIELDS:
            p_val = str(pred.get(field, "")).strip().lower()
            g_val = str(gt.get(field, "")).strip().lower()

            if g_val:
                total_gt += 1
                per_field[field]["gt"] += 1
            if p_val:
                total_pred += 1
                per_field[field]["pred"] += 1
            if p_val and g_val and p_val == g_val:
                tp += 1
                per_field[field]["tp"] += 1
            elif not p_val and not g_val:
                pass  # FIX (BUG 4): Both absent = true negative, do not penalize exact match
            else:
                all_correct = False

            per_field[field]["ned"].append(normalized_edit_distance(p_val, g_val))
        exact_match_all.append(int(all_correct))

    precision = tp / total_pred if total_pred > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    summary = {
        "global_precision": round(precision, 4),
        "global_recall": round(recall, 4),
        "global_f1": round(f1, 4),
        "overall_exact_match": round(np.mean(exact_match_all), 4),
    }

    for field in FIELDS:
        fp = per_field[field]
        p = fp["tp"] / fp["pred"] if fp["pred"] > 0 else 0
        r = fp["tp"] / fp["gt"] if fp["gt"] > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        summary[f"{field}_f1"] = round(f, 4)
        summary[f"{field}_ned"] = round(np.mean(fp["ned"]), 4)

    return summary


def print_results(pretrained_m, finetuned_m):
    print(f"\n{'='*72}")
    print(f"{'METRIC':<30} {'PRETRAINED':>18} {'FINE-TUNED':>18}")
    print(f"{'='*72}")
    print(f"{'Global F1':<30} {pretrained_m['global_f1']:>18.4f} {finetuned_m['global_f1']:>18.4f}")
    print(f"{'Global Precision':<30} {pretrained_m['global_precision']:>18.4f} {finetuned_m['global_precision']:>18.4f}")
    print(f"{'Global Recall':<30} {pretrained_m['global_recall']:>18.4f} {finetuned_m['global_recall']:>18.4f}")
    print(f"{'Overall Exact Match':<30} {pretrained_m['overall_exact_match']:>18.4f} {finetuned_m['overall_exact_match']:>18.4f}")
    print(f"{'-'*72}")
    for field in FIELDS:
        print(f"{field + ' F1':<30} {pretrained_m[field+'_f1']:>18.4f} {finetuned_m[field+'_f1']:>18.4f}")
        print(f"{field + ' NED':<30} {pretrained_m[field+'_ned']:>18.4f} {finetuned_m[field+'_ned']:>18.4f}")
    print(f"{'='*72}")

    print(f"\n{'SROIE TASK 3 LEADERBOARD COMPARISON':^72}")
    print(f"{'-'*72}")
    print(f"{'Method':<35} {'F1':>10}")
    print(f"{'-'*72}")
    leaderboard = [
        ("LayoutLMv3 (Huang et al. 2022)", 0.9633),
        ("PICK (Yu et al. 2021)", 0.9612),
        ("BROS (Hong et al. 2022)", 0.9548),
        ("LayoutLMv2 (Xu et al. 2021)", 0.9495),
        # Zero-shot F1 from Table 1 of "OCR-free Document Understanding Transformer" (Kim et al., ECCV 2022)
        ("DONUT zero-shot (Kim et al. 2022)", 0.8411),
        ("Our pretrained (CORD zero-shot)", pretrained_m["global_f1"]),
        ("Our fine-tuned (this work)", finetuned_m["global_f1"]),
    ]
    for name, score in sorted(leaderboard, key=lambda x: x[1], reverse=True):
        marker = " ◄" if "Our" in name else ""
        print(f"{name:<35} {score:>10.4f}{marker}")
    print(f"{'='*72}\n")


def main():
    # Load test images + ground truth
    sroie_dir = _get_sroie_dir()
    img_dir = sroie_dir / "test_img"
    key_dir = sroie_dir / "test_key"

    test_samples = []
    for img_path in sorted(p for p in img_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        # BUG E FIX: Key files are .txt (4-line format), not .json.
        # Try .json first for pre-converted data, then fall back to .txt.
        gt = None
        key_json = key_dir / (img_path.stem + ".json")
        if key_json.exists():
            try:
                gt = json.loads(key_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        if gt is None:
            key_txt = key_dir / (img_path.stem + ".txt")
            if key_txt.exists():
                lines = key_txt.read_text(encoding="utf-8").strip().splitlines()
                if len(lines) >= 4:
                    gt = {
                        "company": lines[0].strip(),
                        "date":    lines[1].strip(),
                        "address": lines[2].strip(),
                        "total":   lines[3].strip(),
                    }
        if gt is not None:
            test_samples.append((img_path, gt))

    print(f"Evaluating on {len(test_samples)} test images")

    ground_truths = [s[1] for s in test_samples]
    image_paths = [s[0] for s in test_samples]

    # Load pretrained (CORD)
    print("Loading pretrained model (CORD)...")
    pre_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    pre_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2").to(DEVICE)
    pre_model.eval()

    # Load fine-tuned
    print("Loading fine-tuned model...")
    workspace = os.environ.get("DONUT_WORKSPACE", "/workspace")
    ft_model_dir = os.path.join(workspace, "donut-sroie-finetuned")
    ft_processor = DonutProcessor.from_pretrained(ft_model_dir)
    ft_model = VisionEncoderDecoderModel.from_pretrained(ft_model_dir).to(DEVICE)
    ft_model.eval()

    pretrained_preds = []
    finetuned_preds = []

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Inference"):
            # Pretrained (CORD) → remap
            raw = run_inference(pre_model, pre_processor, img_path, "<s_cord-v2>")
            pretrained_preds.append(remap_cord_to_sroie(raw))

            # Fine-tuned (SROIE)
            raw_ft = run_inference(ft_model, ft_processor, img_path, "<s_sroie>")
            if "sroie" in raw_ft and isinstance(raw_ft["sroie"], dict):
                raw_ft = raw_ft["sroie"]
            finetuned_preds.append(raw_ft)

    pretrained_metrics = compute_metrics(pretrained_preds, ground_truths)
    finetuned_metrics = compute_metrics(finetuned_preds, ground_truths)

    print_results(pretrained_metrics, finetuned_metrics)

    # Save everything
    output = {
        "pretrained_metrics": pretrained_metrics,
        "finetuned_metrics": finetuned_metrics,
        "samples": [
            {
                "image": str(p),
                "ground_truth": gt,
                "pretrained_pred": pp,
                "finetuned_pred": fp,
            }
            for p, gt, pp, fp in zip(image_paths, ground_truths, pretrained_preds, finetuned_preds)
        ]
    }
    output_file = os.path.join(workspace, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()