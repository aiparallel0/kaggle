# constants.py — Single source of truth for all shared constants.
#
# Previously, FIELDS, IMAGE_EXTS, MAX_LENGTH, BASE_MODEL, SEED, and NEW_TOKENS
# were duplicated independently in 5+ files (run_experiments.py, train.py,
# evaluate.py, inject_results.py, dataset_loaders.py).  Any change had to be
# replicated manually, risking silent drift.  All files now import from here.

from typing import Dict, FrozenSet, List

# SROIE Task-3 target fields — the four key-value pairs extracted from receipts.
FIELDS: List[str] = ["company", "date", "address", "total"]

# Accepted image file extensions for dataset loading.
IMAGE_EXTS: FrozenSet[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
})

# Maximum decoder token length for DONUT fine-tuning and inference.
# Increased from 512 to 768 to reduce truncation of long address fields,
# which was the weakest-performing SROIE field.
MAX_LENGTH: int = 768

# Base model checkpoint — CORD-pretrained DONUT used as starting point.
BASE_MODEL: str = "naver-clova-ix/donut-base-finetuned-cord-v2"

# Global random seed for reproducibility across all experiments.
SEED: int = 42

# Special tokens added to the tokenizer for SROIE structured output.
NEW_TOKENS: List[str] = [
    "<s_sroie>", "</s_sroie>",
    "<s_company>", "</s_company>",
    "<s_date>",    "</s_date>",
    "<s_address>", "</s_address>",
    "<s_total>",   "</s_total>",
]

# Empty ground-truth template matching the SROIE schema.
EMPTY_GT: Dict[str, str] = {"company": "", "date": "", "address": "", "total": ""}
