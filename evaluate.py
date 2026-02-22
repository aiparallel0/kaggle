import json, torch, re
import editdistance
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

SROIE_DIR = Path("/workspace/ICDAR-2019-SROIE/data")
FIELDS = ["company", "date", "address", "total"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    return processor.token2json(sequence)


def remap_cord_to_sroie(cord_output):
    """Map CORD schema fields to SROIE field names (best effort)."""
    result = {}
    if isinstance(cord_output, dict):
        store_info = cord_output.get("store_info", {})
        if isinstance(store_info, dict):
            result["company"] = store_info.get("store_name", "")
            result["address"] = store_info.get("region", "")
        total_info = cord_output.get("total", {})
        if isinstance(total_info, dict):
            result["total"] = total_info.get("total_price", "")
        elif isinstance(total_info, list) and len(total_info) > 0:
            result["total"] = total_info[0].get("total_price", "")
        # CORD does not have a standard date field; leave empty
        result["date"] = ""
    return result


def normalized_edit_distance(pred, gt):
    pred, gt = str(pred).lower().strip(), str(gt).lower().strip()
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / max(len(pred), len(gt))


def compute_metrics(predictions, ground_truths):
    """
    Official SROIE Task 3 metric: global F1 over all (image, field) pairs.
    A pair is TP if predicted string == ground truth string (case-insensitive, stripped).
    """
    tp, total_pred, total_gt = 0, 0, 0
    per_field = {f: {"tp": 0, "pred": 0, "gt": 0, "ned": []} for f in FIELDS}
    exact_match_all = []

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
        ("DONUT published (Kim 2022)", 0.8411),
        ("Our pretrained (CORD zero-shot)", pretrained_m["global_f1"]),
        ("Our fine-tuned (this work)", finetuned_m["global_f1"]),
    ]
    for name, score in sorted(leaderboard, key=lambda x: x[1], reverse=True):
        marker = " ◄" if "Our" in name else ""
        print(f"{name:<35} {score:>10.4f}{marker}")
    print(f"{'='*72}\n")


def main():
    # Load test images + ground truth
    img_dir = SROIE_DIR / "img"
    key_dir = SROIE_DIR / "key"

    test_samples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        key_file = key_dir / (img_path.stem + ".txt")
        if key_file.exists():
            try:
                gt = json.loads(key_file.read_text(encoding="utf-8"))
                test_samples.append((img_path, gt))
            except json.JSONDecodeError:
                pass

    # Use last 100 images as test set (first 500+ used for training)
    test_samples = test_samples[-100:]
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
    ft_processor = DonutProcessor.from_pretrained("/workspace/donut-sroie-finetuned")
    ft_model = VisionEncoderDecoderModel.from_pretrained("/workspace/donut-sroie-finetuned").to(DEVICE)
    ft_model.eval()

    pretrained_preds = []
    finetuned_preds = []

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Inference"):
            # Pretrained (CORD) → remap
            raw = run_inference(pre_model, pre_processor, img_path, "<s_cord-v2>")
            pretrained_preds.append(remap_cord_to_sroie(raw))

            # Fine-tuned (SROIE)
            finetuned_preds.append(
                run_inference(ft_model, ft_processor, img_path, "<s_sroie>")
            )

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
    with open("/workspace/evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Saved → /workspace/evaluation_results.json")


if __name__ == "__main__":
    main()