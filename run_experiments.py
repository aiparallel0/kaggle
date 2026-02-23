"""
run_experiments.py — Experiment orchestrator for 8 DONUT fine-tuning experiments.

Usage
-----
Run all experiments:
    python run_experiments.py --all

Run a single experiment (e.g., experiment 2):
    python run_experiments.py --experiment 2

Each experiment trains DONUT from the CORD checkpoint and evaluates on the
SROIE test set (100 images in test_img / test_key).  Results are saved to
results/experiment_N.json and a summary to results/all_experiments.json.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor, VisionEncoderDecoderModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

import dataset_loaders
from evaluate import compute_metrics, run_inference

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")

BASE_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"
WORKSPACE = Path(os.environ.get("DONUT_WORKSPACE", "/workspace"))
FIELDS = ["company", "date", "address", "total"]
MAX_LENGTH = 512
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NOTE: DEVICE is evaluated at import time. Both run_experiments.py and evaluate.py
# define this constant independently; they will agree as long as GPU availability
# does not change between module imports, which is the expected runtime assumption.
SEED = 42

# Hyperparameter config — stored in each result JSON for cache validation
TRAIN_CONFIG = {
    "num_train_epochs": 5,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 4,
    "base_model": BASE_MODEL,
}

NEW_TOKENS = [
    "<s_sroie>", "</s_sroie>",
    "<s_company>", "</s_company>",
    "<s_date>",    "</s_date>",
    "<s_address>", "</s_address>",
    "<s_total>",   "</s_total>",
]


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS: Dict[int, Dict] = {
    1: {
        "name": "SROIE only (baseline)",
        "datasets": ["sroie"],
        "description": "Fine-tune on SROIE training set only (526 images).",
    },
    2: {
        "name": "SROIE + WildReceipt",
        "datasets": ["sroie", "wildreceipt"],
        "description": "Add ~1 740 WildReceipt receipt images with KIE remapping.",
    },
    3: {
        "name": "SROIE + SROIE-NER",
        "datasets": ["sroie", "sroie_ner"],
        "description": "Add SROIE in token-level NER format (different augmentation view).",
    },
    4: {
        "name": "SROIE + CORD",
        "datasets": ["sroie", "cord"],
        "description": "Add ~900 CORD receipt images (model's original pretraining data).",
    },
    5: {
        "name": "SROIE + WildReceipt + CORD",
        "datasets": ["sroie", "wildreceipt", "cord"],
        "description": "Combine SROIE, WildReceipt, and CORD.",
    },
    6: {
        "name": "SROIE + SROIE-NER + CORD",
        "datasets": ["sroie", "sroie_ner", "cord"],
        "description": "Combine SROIE, SROIE-NER, and CORD.",
    },
    7: {
        "name": "SROIE + All",
        "datasets": ["sroie", "wildreceipt", "sroie_ner", "cord"],
        "description": "Combine all four available datasets.",
    },
    8: {
        "name": "SROIE + Invoices-DONUT",
        "datasets": ["sroie", "invoices_donut"],
        "description": "Add ~800 structured invoice images (cross-domain transfer from invoices to receipts).",
    },
}

# ---------------------------------------------------------------------------
# PyTorch Dataset that works from a list of (image_path, gt_dict) tuples
# ---------------------------------------------------------------------------

class MultiDataset(Dataset):
    """Wraps a list of (Path, dict) samples into a PyTorch Dataset."""

    def __init__(self, samples: List[Tuple[Path, Dict]], processor, max_length: int = MAX_LENGTH):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, gt = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        target = "<s_sroie>"
        for f in FIELDS:
            v = gt.get(f, "")
            target += f"<s_{f}>{v}</s_{f}>"
        target += "</s_sroie>"

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            target, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_experiment(exp_id: int, samples: List[Tuple[Path, Dict]], output_dir: Path) -> None:
    """Fine-tune DONUT on *samples* and save the model to *output_dir*."""
    set_seed()
    print(f"\n[Exp {exp_id}] Training on {len(samples)} samples → {output_dir}")

    processor = DonutProcessor.from_pretrained(BASE_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    processor.tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    model.gradient_checkpointing_enable()

    train_ds = MultiDataset(samples, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        warmup_steps=100,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"[Exp {exp_id}] Model saved → {output_dir}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_experiment(exp_id: int, model_dir: Path) -> Dict:
    """Evaluate a fine-tuned model (at *model_dir*) on the SROIE test set."""
    test_samples = dataset_loaders.load_sroie_test()
    print(f"[Exp {exp_id}] Evaluating on {len(test_samples)} SROIE test images")

    ground_truths = [s[1] for s in test_samples]
    image_paths = [s[0] for s in test_samples]

    # Load fine-tuned model
    ft_processor = DonutProcessor.from_pretrained(str(model_dir))
    ft_model = VisionEncoderDecoderModel.from_pretrained(str(model_dir)).to(DEVICE)
    ft_model.eval()

    finetuned_preds = []
    empty_count = 0
    with torch.no_grad():
        for img_path in image_paths:
            pred = run_inference(ft_model, ft_processor, img_path, "<s_sroie>")
            if "sroie" in pred and isinstance(pred["sroie"], dict):
                pred = pred["sroie"]
            if not pred:
                empty_count += 1
            finetuned_preds.append(pred)

    # FIX (BUG 5): Report how many predictions were empty so users can detect
    # systematic token2json failures that would silently contaminate metrics.
    if empty_count > 0:
        print(
            f"[Exp {exp_id}] WARNING: {empty_count} of {len(image_paths)} predictions were empty"
            " (token2json failure or empty model output)"
        )

    metrics = compute_metrics(finetuned_preds, ground_truths)
    return metrics


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(exp_id: int) -> Dict:
    """Run a single experiment: train, evaluate, save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment ID {exp_id}. Valid: {list(EXPERIMENTS)}")

    exp = EXPERIMENTS[exp_id]
    print(f"\n{'='*72}")
    print(f"Experiment {exp_id}: {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"Datasets: {exp['datasets']}")
    print(f"{'='*72}")

    result_file = RESULTS_DIR / f"experiment_{exp_id}.json"

    # Check if already done - validate cached result matches current experiment
    # definition (datasets AND hyperparameters) before reusing.
    if result_file.exists():
        with open(result_file) as fh:
            cached = json.load(fh)
        if cached.get("datasets") != exp["datasets"] or cached.get("config") != TRAIN_CONFIG:
            # NOTE: JSON round-trip preserves numeric equality for floats like 5e-5,
            # so this comparison is safe (5e-5 == 5e-05 after json.load).
            print(
                f"[Exp {exp_id}] STALE result detected (datasets or config mismatch). "
                "Deleting and re-running."
            )
            result_file.unlink()
        else:
            print(f"[Exp {exp_id}] Valid cached result found — skipping.")
            return cached

    # Load data
    samples = dataset_loaders.get_combined_dataset(exp["datasets"])
    if len(samples) == 0:
        print(f"[Exp {exp_id}] WARNING: No samples loaded — saving empty result.")
        result = {
            "experiment_id": exp_id,
            "name": exp["name"],
            "datasets": exp["datasets"],
            "config": TRAIN_CONFIG,
            "num_train_samples": 0,
            "metrics": {},
            "error": "No training samples available",
        }
        result_file.write_text(json.dumps(result, indent=2))
        return result

    # Train
    model_dir = WORKSPACE / "models" / f"experiment_{exp_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_experiment(exp_id, samples, model_dir)

    # Evaluate
    metrics = evaluate_experiment(exp_id, model_dir)

    # Save result
    result = {
        "experiment_id": exp_id,
        "name": exp["name"],
        "datasets": exp["datasets"],
        "config": TRAIN_CONFIG,
        "num_train_samples": len(samples),
        "metrics": metrics,
    }
    result_file.write_text(json.dumps(result, indent=2))
    print(f"[Exp {exp_id}] Results saved → {result_file}")
    print(f"[Exp {exp_id}] Global F1 = {metrics.get('global_f1', 'N/A')}")
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def save_summary() -> None:
    """Collect all individual result files into results/all_experiments.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for exp_id in EXPERIMENTS:
        result_file = RESULTS_DIR / f"experiment_{exp_id}.json"
        if result_file.exists():
            with open(result_file) as fh:
                all_results[str(exp_id)] = json.load(fh)

    summary_file = RESULTS_DIR / "all_experiments.json"
    summary_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nSummary saved → {summary_file}")

    # Pretty-print leaderboard
    print(f"\n{'='*72}")
    print(f"{'Exp':<5} {'Name':<35} {'Train Samples':>14} {'Global F1':>10}")
    print(f"{'-'*72}")
    for exp_id_str, res in sorted(all_results.items(), key=lambda x: int(x[0])):
        name = res.get("name", "")[:34]
        n = res.get("num_train_samples", 0)
        f1 = res.get("metrics", {}).get("global_f1", float("nan"))
        import math
        f1_str = f"{f1:>10.4f}" if not math.isnan(f1) else "       N/A"
        print(f"{exp_id_str:<5} {name:<35} {n:>14} {f1_str}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run DONUT SROIE experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all 8 experiments sequentially")
    group.add_argument("--experiment", type=int, metavar="N",
                       choices=range(1, len(EXPERIMENTS) + 1),
                       help="Run a single experiment (1-8)")
    # FIX (BUG 3): --force deletes all cached result files so every experiment
    # is re-run from scratch regardless of cached state.
    parser.add_argument("--force", action="store_true",
                        help="Delete all cached results and re-run from scratch")
    args = parser.parse_args()

    if args.force:
        for result_file in RESULTS_DIR.glob("experiment_*.json"):
            result_file.unlink()
            print(f"[force] Deleted cached result: {result_file}")

    if args.all:
        for exp_id in EXPERIMENTS:
            run_experiment(exp_id)
        save_summary()
    else:
        run_experiment(args.experiment)
        save_summary()


if __name__ == "__main__":
    main()
