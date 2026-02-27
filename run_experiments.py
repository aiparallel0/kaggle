# MIT License
#
# Copyright (c) 2024
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
run_experiments.py — Experiment orchestrator for 8 DONUT fine-tuning experiments.

Usage
-----
Run all experiments:
    python run_experiments.py --all

Run a single experiment (e.g., experiment 2):
    python run_experiments.py --experiment 2

Force re-run (ignore cached results):
    python run_experiments.py --all --force

Each experiment trains DONUT from the CORD checkpoint and evaluates on the
SROIE test set (347 images in test_img / test_key).  Results are saved to
results/experiment_N.json and a summary to results/all_experiments.json.

Architecture
------------
ExperimentConfig is THE single source of truth for all training hyperparameters.
No hardcoded epoch values, learning rates, or batch sizes exist outside of it.
DonutTrainer (from train.py) receives an ExperimentConfig and reads all
hyperparameters from it via duck-typed attribute access.
DonutEvaluator (from evaluate.py) handles model loading with weight re-tying
and evaluation with self-test and parse-failure thresholds.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel

import dataset_loaders
from evaluate import compute_metrics, run_inference

# FIX: Import shared constants from single source of truth (constants.py)
# instead of duplicating FIELDS/IMAGE_EXTS/etc. independently in this file.
from constants import FIELDS, MAX_LENGTH, IMAGE_EXTS, NEW_TOKENS, BASE_MODEL, SEED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")

WORKSPACE = Path(os.environ.get("DONUT_WORKSPACE", "/workspace"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
# ExperimentConfig — THE single source of truth for all hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Single source of truth for all training hyperparameters.

    Every training parameter (epochs, lr, batch_size, etc.) lives here.
    DonutTrainer reads them via duck-typed attribute access using the
    property aliases below (max_epochs, learning_rate, etc.).
    """

    name: str
    datasets: List[str]
    epochs: int = 30
    lr: float = 5e-5
    batch_size: int = 16
    seed: int = 42
    early_stopping_patience: int = 3
    base_model: str = "naver-clova-ix/donut-base-finetuned-cord-v2"
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_length: int = 512
    gradient_accumulation_steps: int = 2
    description: str = ""
    experiment_id: int = 0

    # -- Duck-typed aliases for DonutTrainer compatibility ----------------
    # DonutTrainer reads config.max_epochs, config.learning_rate, etc.
    # These properties ensure a single source of truth (no duplication).

    @property
    def max_epochs(self) -> int:
        return self.epochs

    @property
    def learning_rate(self) -> float:
        return self.lr

    @property
    def per_device_train_batch_size(self) -> int:
        return self.batch_size

    @property
    def output_dir(self) -> str:
        """Default output directory; overridden at call site when needed."""
        return str(WORKSPACE / "models" / f"experiment_{self.experiment_id}")


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS: Dict[int, ExperimentConfig] = {
    1: ExperimentConfig(
        name="SROIE only (baseline)",
        datasets=["sroie"],
        description="Fine-tune on SROIE training set only.",
        experiment_id=1,
    ),
    2: ExperimentConfig(
        name="SROIE + WildReceipt",
        datasets=["sroie", "wildreceipt"],
        description="Add ~1 740 WildReceipt receipt images with KIE remapping.",
        experiment_id=2,
    ),
    3: ExperimentConfig(
        name="SROIE + Invoices-DONUT",
        datasets=["sroie", "invoices_donut"],
        description="Add Invoices-DONUT invoice images.",
        experiment_id=3,
    ),
    4: ExperimentConfig(
        name="SROIE + CORD",
        datasets=["sroie", "cord"],
        description="Add ~900 CORD receipt images (model's original pretraining data).",
        experiment_id=4,
    ),
    5: ExperimentConfig(
        name="SROIE + WildReceipt + CORD",
        datasets=["sroie", "wildreceipt", "cord"],
        description="Combine SROIE, WildReceipt, and CORD.",
        experiment_id=5,
    ),
    6: ExperimentConfig(
        name="SROIE + WildReceipt + Invoices",
        datasets=["sroie", "wildreceipt", "invoices_donut"],
        description="Combine SROIE, WildReceipt, and Invoices-DONUT.",
        experiment_id=6,
    ),
    7: ExperimentConfig(
        name="SROIE + CORD + Invoices",
        datasets=["sroie", "cord", "invoices_donut"],
        description="Combine SROIE, CORD, and Invoices-DONUT.",
        experiment_id=7,
    ),
    8: ExperimentConfig(
        name="SROIE + All",
        datasets=["sroie", "wildreceipt", "cord", "invoices_donut"],
        description="Combine all four available datasets.",
        experiment_id=8,
    ),
}

# ---------------------------------------------------------------------------
# TRAIN_CONFIG — cache validation for result JSON files
# ---------------------------------------------------------------------------
# FIX: Previously only included 5 of 9 hyperparameters (max_epochs, lr,
# batch_size, early_stopping_patience, base_model).  If warmup_steps,
# weight_decay, max_length, or seed changed, the cache validation at
# cached.get("config") != TRAIN_CONFIG would NOT detect stale results.
# Now includes ALL hyperparameters from ExperimentConfig for complete
# staleness detection.

_default_config = ExperimentConfig(name="", datasets=[])

TRAIN_CONFIG: Dict[str, Any] = {
    "max_epochs": _default_config.epochs,
    "learning_rate": _default_config.lr,
    "per_device_train_batch_size": _default_config.batch_size,
    "early_stopping_patience": _default_config.early_stopping_patience,
    "base_model": _default_config.base_model,
    # FIX: These were previously missing, causing stale cache hits when
    # warmup_steps/weight_decay/max_length/seed changed.
    "warmup_steps": _default_config.warmup_steps,
    "weight_decay": _default_config.weight_decay,
    "max_length": _default_config.max_length,
    "seed": _default_config.seed,
    "gradient_accumulation_steps": _default_config.gradient_accumulation_steps,
}


# ---------------------------------------------------------------------------
# PyTorch Dataset that works from a list of (image_path, gt_dict) tuples
# ---------------------------------------------------------------------------

class MultiDataset(Dataset):
    """Wraps a list of (Path, dict) samples into a PyTorch Dataset.

    When sufficient RAM is available, pre-loads all images into memory
    to eliminate disk I/O during training.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, Dict]],
        processor: DonutProcessor,
        max_length: int = MAX_LENGTH,
        cache_in_ram: bool = True,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self._image_cache: Dict[int, Image.Image] = {}

        if cache_in_ram and len(samples) > 0:
            # Estimate memory: ~3MB per receipt image × num_samples
            estimated_mb = len(samples) * 3
            try:
                import psutil
                available_mb = psutil.virtual_memory().available // (1024 * 1024)
            except ImportError:
                available_mb = 0  # skip caching if psutil unavailable

            # Only cache if we'd use less than 50% of available RAM
            if available_mb > 0 and estimated_mb < available_mb * 0.5:
                print(f"  [RAM Cache] Pre-loading {len(samples)} images into RAM "
                      f"(~{estimated_mb}MB / {available_mb}MB available) ...")
                import concurrent.futures

                def _load_one(idx_path):
                    idx, path = idx_path
                    try:
                        return idx, Image.open(path).convert("RGB")
                    except Exception:
                        return idx, None

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    for idx, img in pool.map(_load_one, enumerate(
                            (s[0] for s in samples))):
                        if img is not None:
                            self._image_cache[idx] = img

                print(f"  [RAM Cache] {len(self._image_cache)}/{len(samples)} images cached")
            else:
                if available_mb > 0:
                    print(f"  [RAM Cache] Skipped (need ~{estimated_mb}MB, "
                          f"available {available_mb}MB)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, gt = self.samples[idx]

        # Use cached image if available, otherwise load from disk
        if idx in self._image_cache:
            image = self._image_cache[idx]
        else:
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
# Training — delegates to DonutTrainer from train.py
# ---------------------------------------------------------------------------

def train_experiment(
    exp_id: int,
    samples: List[Tuple[Path, Dict]],
    output_dir: Path,
    val_samples: List[Tuple[Path, Dict]] = None,
) -> List[Dict]:
    """Fine-tune DONUT on *samples* and save the model to *output_dir*.

    All hyperparameters come from ``EXPERIMENTS[exp_id]`` (an ExperimentConfig).
    Training is delegated to ``DonutTrainer`` from ``train.py``, which reads
    hyperparameters from the config via duck-typed attributes.

    Returns the trainer log history (list of per-step dicts) for convergence
    plot generation.
    """
    from train import DonutTrainer

    config = EXPERIMENTS[exp_id]

    set_seed(config.seed)
    print(f"\n[Exp {exp_id}] Training on {len(samples)} samples → {output_dir}")
    print(f"[Exp {exp_id}] Hyperparams: epochs={config.epochs}, "
          f"lr={config.lr}, batch_size={config.batch_size}, "
          f"warmup={config.warmup_steps}, wd={config.weight_decay}")
    if val_samples:
        print(f"[Exp {exp_id}] Validation set: {len(val_samples)} samples")

    # Load base model and processor
    processor = DonutProcessor.from_pretrained(config.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(config.base_model)

    # Add SROIE special tokens
    processor.tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # After resize, embed_tokens and lm_head are separate tensors with
    # independent random init for the new tokens.  Set tie_word_embeddings=False
    # so save_pretrained() saves BOTH weights independently.  Without this,
    # the saved checkpoint omits lm_head (or tie_weights() overwrites the
    # learned lm_head with embed_tokens), causing F1=0 on reload.
    model.decoder.config.tie_word_embeddings = False

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        ["<s_sroie>"]
    )[0]
    model.gradient_checkpointing_enable()

    # Build PyTorch datasets
    train_ds = MultiDataset(samples, processor, max_length=config.max_length)
    val_ds = MultiDataset(val_samples, processor, max_length=config.max_length) if val_samples else None

    # Verify single source of truth: ExperimentConfig properties map correctly
    assert config.epochs == config.max_epochs, (
        f"Single source of truth violation: epochs={config.epochs} "
        f"!= max_epochs={config.max_epochs}"
    )
    assert config.lr == config.learning_rate, (
        f"Single source of truth violation: lr={config.lr} "
        f"!= learning_rate={config.learning_rate}"
    )
    assert config.batch_size == config.per_device_train_batch_size, (
        f"Single source of truth violation: batch_size={config.batch_size} "
        f"!= per_device_train_batch_size={config.per_device_train_batch_size}"
    )

    # Create DonutTrainer — it reads all hyperparams from config
    trainer = DonutTrainer(
        config=config,
        processor=processor,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    # Train
    result = trainer.train()

    # Save model with tied weights
    trainer.save(output_dir)

    print(f"[Exp {exp_id}] Training complete "
          f"(duration={result.duration_seconds:.1f}s, "
          f"train={result.train_samples}, val={result.val_samples})")

    # FIX: Explicit GPU cleanup between experiments to prevent OOM on GPUs
    # with limited VRAM.  The RTX 4090 has 24GB — sufficient for DONUT but
    # running 8+ experiments sequentially without cleanup risks fragmentation.
    import gc
    del model, processor, trainer, train_ds
    if val_ds is not None:
        del val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Exp {exp_id}] GPU memory released")

    return result.log_history


# ---------------------------------------------------------------------------
# Evaluation — delegates to DonutEvaluator from evaluate.py
# ---------------------------------------------------------------------------

def evaluate_experiment(exp_id: int, model_dir: Path) -> Dict:
    """Evaluate a fine-tuned model (at *model_dir*) on the SROIE test set.

    Uses DonutEvaluator from evaluate.py which handles:
      - Weight re-tying via load_model_with_tied_weights (fixes lm_head bug)
      - Self-test before full evaluation
      - Parse failure threshold checking
    """
    from evaluate import DonutEvaluator, load_model_with_tied_weights

    config = EXPERIMENTS[exp_id]
    test_samples = dataset_loaders.load_sroie_test()
    print(f"[Exp {exp_id}] Evaluating on {len(test_samples)} SROIE test images")

    # Load processor from the fine-tuned model directory
    processor = DonutProcessor.from_pretrained(str(model_dir))

    # Create evaluator — handles model loading with weight re-tying
    evaluator = DonutEvaluator(
        model_path=model_dir,
        processor=processor,
        test_dataset=test_samples,
        task_prompt="<s_sroie>",
        max_length=config.max_length,
        device=DEVICE,
    )

    # Run evaluation (includes self-test + parse failure threshold)
    eval_result = evaluator.evaluate()
    metrics = eval_result.to_dict()

    if eval_result.parse_failures > 0:
        print(
            f"[Exp {exp_id}] WARNING: {eval_result.parse_failures} of "
            f"{eval_result.num_samples} predictions had parse failures"
        )

    return metrics


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(exp_id: int) -> Dict:
    """Run a single experiment: train, evaluate, save results.

    Checks cache validity (datasets AND hyperparams must match) before
    reusing a previous result.  Loads data via dataset_loaders, delegates
    training to train_experiment and evaluation to evaluate_experiment,
    then saves the result JSON.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment ID {exp_id}. Valid: {list(EXPERIMENTS)}")

    config = EXPERIMENTS[exp_id]
    print(f"\n{'='*72}")
    print(f"Experiment {exp_id}: {config.name}")
    print(f"Description: {config.description}")
    print(f"Datasets: {config.datasets}")
    print(f"{'='*72}")

    result_file = RESULTS_DIR / f"experiment_{exp_id}.json"

    # Check if already done — validate cached result matches current experiment
    # definition (datasets AND hyperparameters) before reusing.
    if result_file.exists():
        with open(result_file) as fh:
            cached = json.load(fh)
        if cached.get("datasets") != config.datasets or cached.get("config") != TRAIN_CONFIG:
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
    train_samples, val_samples = dataset_loaders.get_combined_dataset(config.datasets)
    if len(train_samples) == 0:
        print(f"[Exp {exp_id}] WARNING: No samples loaded — saving empty result.")
        result = {
            "experiment_id": exp_id,
            "name": config.name,
            "datasets": config.datasets,
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
    log_history = train_experiment(exp_id, train_samples, model_dir, val_samples=val_samples)

    # Evaluate
    metrics = evaluate_experiment(exp_id, model_dir)

    # Save result
    result = {
        "experiment_id": exp_id,
        "name": config.name,
        "datasets": config.datasets,
        "config": TRAIN_CONFIG,
        "num_train_samples": len(train_samples),
        "metrics": metrics,
        "training_log": log_history,
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
