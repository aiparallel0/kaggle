# Re-export shim — logic moved to data_pipeline.py
# Kept for backward compatibility with DO NOT TOUCH imports in:
#   run_experiments.py, dataset_preparation.py
#
# ── Transformers compat shim (Pattern 3, CLAUDE.md §5) ──────────────────
# In transformers ≥4.47 PreTrainedTokenizerBase moved to
# transformers.tokenization_utils_base.  The `datasets` library still
# tries to access it at the old location during load_dataset() which
# causes an AttributeError.  Restore the alias so CORD and
# Invoices-DONUT downloads succeed.
try:
    import transformers as _transformers

    if not hasattr(_transformers, "PreTrainedTokenizerBase"):
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        _transformers.PreTrainedTokenizerBase = PreTrainedTokenizerBase
except Exception:
    pass

from constants import _get_sroie_dir  # noqa: F401 (private re-export for tests)
from data_pipeline import *  # noqa: F401, F403
from data_pipeline import (
    _get_datasets_dir,  # noqa: F401 (private re-export for tests)
    _load_key_file,  # noqa: F401 (private re-export for train_trocr_yolo.py)
    _load_seller_split_cache,  # noqa: F401 (private re-export for tests)
)
