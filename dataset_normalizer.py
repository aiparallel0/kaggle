# Re-export shim — dataset normalizer merged into data_pipeline.py
# Kept for backward compatibility with tests that import from dataset_normalizer.
from data_pipeline import (  # noqa: F401
    DatasetNormalizer,
    extract_address_from_seller,
    normalise_samples,
)
