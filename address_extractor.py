"""address_extractor.py — Utility to split a combined seller/company string.

The Invoices-DONUT dataset stores both the company name and the address as a
single combined ``seller`` string, e.g.:

    "Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228"

``extract_address_from_seller`` uses regex heuristics to detect where the
address portion begins and splits the string into (company_name, address).
If no address pattern is detected the full string is returned as the company
name with an empty address.
"""

from __future__ import annotations

import re

__all__ = ["extract_address_from_seller"]

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Matches a leading street number (one or more digits) followed by a space and
# at least one letter — e.g. "123 Main", "4500 Oak".
_STREET_NUMBER_RE = re.compile(r"\b\d+\s+[A-Za-z]")

# P.O. Box variants
_PO_BOX_RE = re.compile(r"\bP\.?\s*O\.?\s*Box\b", re.IGNORECASE)

# Common street-type suffixes that follow a street name.
_STREET_SUFFIX_RE = re.compile(
    r"\b(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|"
    r"Way|Lane|Ln|Court|Ct|Place|Pl|Terrace|Ter|Circle|Cir|"
    r"Highway|Hwy|Parkway|Pkwy|Trail|Trl|Run|Loop|Row)\b",
    re.IGNORECASE,
)

# US-style state abbreviation + ZIP (e.g. "MA 46228" or "CA 90210-1234").
_STATE_ZIP_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")

# Compiled list of patterns ordered by decreasing specificity.
_ADDRESS_PATTERNS: list[re.Pattern[str]] = [
    _PO_BOX_RE,
    _STREET_NUMBER_RE,
    _STREET_SUFFIX_RE,
    _STATE_ZIP_RE,
]


def extract_address_from_seller(seller_str: str) -> tuple[str, str]:
    """Split a combined seller string into (company_name, address).

    Parameters
    ----------
    seller_str:
        Raw seller string from the Invoices-DONUT dataset, e.g.
        ``"Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228"``

    Returns
    -------
    tuple[str, str]
        ``(company_name, address)`` where *address* may be an empty string when
        no address pattern is detected.
    """
    seller_str = seller_str.strip()
    if not seller_str:
        return ("", "")

    # Find the earliest position in the string where any address pattern matches.
    earliest_start: int | None = None

    for pattern in _ADDRESS_PATTERNS:
        m = pattern.search(seller_str)
        if m and (earliest_start is None or m.start() < earliest_start):
            earliest_start = m.start()

    if earliest_start is None:
        # No address detected — return full string as company, empty address.
        return (seller_str, "")

    company_name = seller_str[:earliest_start].strip().rstrip(",").strip()
    address = seller_str[earliest_start:].strip()

    return (company_name, address)
