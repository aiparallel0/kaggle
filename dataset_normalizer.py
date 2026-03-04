"""
dataset_normalizer.py — Canonical intermediate schema enforcer.

Sits between dataset loaders and the training pipeline.  Every dataset
loader (SROIELoader, WildReceiptLoader, InvoicesDonutLoader, etc.) produces
(Path, dict) samples in its own native format.  This module normalises them
to the SROIE canonical schema before they reach the model.

This breaks the historical cycle where every new dataset packaging variant
required a new ad-hoc fix inside an individual loader.

Canonical schema
----------------
  {
      "company": str,   # store / merchant name; "" if absent
      "date":    str,   # receipt date; "" if absent
      "address": str,   # store address; "" if absent
      "total":   str,   # grand total amount; "" if absent
  }

All four keys are ALWAYS present.  Values are stripped of leading/trailing
whitespace.  No key is ever None or absent.
"""

from __future__ import annotations

import logging
from pathlib import Path

from constants import FIELDS

logger = logging.getLogger(__name__)

# Field alias map: non-canonical name → canonical SROIE field name
FIELD_ALIASES: dict[str, str] = {
    # company aliases
    "store_name": "company",
    "merchant": "company",
    "vendor": "company",
    "shop": "company",
    "seller": "company",
    "business_name": "company",
    # date aliases
    "receipt_date": "date",
    "transaction_date": "date",
    "invoice_date": "date",
    # total aliases
    "amount": "total",
    "grand_total": "total",
    "total_amount": "total",
    "sum": "total",
    # address aliases
    "store_address": "address",
    "location": "address",
}

# Currency prefixes to strip during normalisation (only used for YOLO/TrOCR pipeline).
# Order matters: longer prefixes (e.g. "S$") must appear before any prefix they start
# with (e.g. "$") so that the longest match is found first.
_CURRENCY_PREFIXES = ("RM", "S$", "USD", "SGD", "MYR", "$", "£", "€", "¥")

Sample = tuple[Path, dict[str, str]]


class DatasetNormalizer:
    """Normalise a list of (Path, dict) samples to the canonical SROIE schema.

    Parameters
    ----------
    coverage_threshold:
        Minimum fraction of samples that must have a non-empty value for each
        of the *mandatory* fields (company, date, total).  Raises ValueError
        if coverage drops below this threshold after normalisation.
        Set to 0.0 to disable the check.
    strip_currency:
        If True, strip common currency symbols from the 'total' field.
        Use for the YOLO/TrOCR pipeline only — NOT for DONUT (which learns
        the raw format including currency symbols).
    source_name:
        Human-readable dataset name for log messages.
    """

    MANDATORY_FIELDS = ("company", "date", "total")

    def __init__(
        self,
        coverage_threshold: float = 0.30,
        strip_currency: bool = False,
        source_name: str = "unknown",
    ) -> None:
        self.coverage_threshold = coverage_threshold
        self.strip_currency = strip_currency
        self.source_name = source_name

    def normalise(self, samples: list[Sample]) -> list[Sample]:
        """Return a new list of samples normalised to the canonical schema."""
        if not samples:
            return []

        normalised: list[Sample] = []
        for img_path, raw_gt in samples:
            gt = self._normalise_dict(raw_gt)
            normalised.append((img_path, gt))

        self._assert_coverage(normalised)
        return normalised

    def _normalise_dict(self, raw: dict) -> dict[str, str]:
        """Normalise a single ground-truth dict to the canonical schema."""
        # Step 1: resolve aliases
        resolved: dict[str, str] = {}
        for k, v in raw.items():
            normalised_key = k.lower().strip()
            canonical_key = FIELD_ALIASES.get(normalised_key, normalised_key)
            if canonical_key in FIELDS:
                resolved[canonical_key] = str(v).strip() if v is not None else ""

        # Step 2: ensure all 4 canonical keys are present, default to ""
        result: dict[str, str] = {f: "" for f in FIELDS}
        for f in FIELDS:
            val = resolved.get(f, "")
            # Treat "N/A", "n/a", "null", "none", "nan" as empty
            if val.lower() in ("n/a", "null", "none", "nan", "-"):
                val = ""
            result[f] = val

        # Step 3: optional currency strip (YOLO/TrOCR only)
        if self.strip_currency and result["total"]:
            total = result["total"].strip()
            for prefix in _CURRENCY_PREFIXES:
                if total.upper().startswith(prefix.upper()):
                    total = total[len(prefix) :].strip()
                    break
            result["total"] = total

        return result

    def _assert_coverage(self, samples: list[Sample]) -> None:
        """Raise ValueError if coverage for any mandatory field is below threshold."""
        if self.coverage_threshold <= 0.0 or not samples:
            return
        n = len(samples)
        for field in self.MANDATORY_FIELDS:
            count = sum(1 for _, gt in samples if gt.get(field, ""))
            coverage = count / n
            if coverage < self.coverage_threshold:
                raise ValueError(
                    f"[DatasetNormalizer] {self.source_name}: field '{field}' "
                    f"coverage {coverage:.1%} is below threshold {self.coverage_threshold:.1%} "
                    f"({count}/{n} samples have non-empty values). "
                    f"Check the dataset loader or lower coverage_threshold."
                )
            logger.info(
                "[DatasetNormalizer] %s: field '%s' coverage %.1f%% (%d/%d)",
                self.source_name,
                field,
                coverage * 100,
                count,
                n,
            )


def normalise_samples(
    samples: list[Sample],
    source_name: str = "unknown",
    coverage_threshold: float = 0.30,
    strip_currency: bool = False,
) -> list[Sample]:
    """Convenience wrapper around DatasetNormalizer.normalise()."""
    return DatasetNormalizer(
        coverage_threshold=coverage_threshold,
        strip_currency=strip_currency,
        source_name=source_name,
    ).normalise(samples)
