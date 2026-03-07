# =============================================================================
# preprocess_seller_split.py
# Purpose: Seller-stratified train/val split to prevent data leakage across sellers
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-07
# =============================================================================
"""preprocess_seller_split.py — Split Invoices-DONUT seller strings into
(company, address) using spaCy NER, with heuristic fallback.

The Invoices-DONUT dataset stores both the company name and the mailing
address in a single ``seller`` field, e.g.:

    "Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228"

This script:
1. Loads all ground_truth strings from the ``katanaml-org/invoices-donut-data-v1``
   HuggingFace dataset.
2. Splits each seller string using spaCy NER (ORG/PERSON entities → company;
   GPE/LOC/FAC → address parts).
3. Falls back to improved heuristics (street-number regex after the last
   ORG/PERSON entity span) when NER entity boundaries are ambiguous.
4. Saves results to ``seller_split_cache.json`` (keyed by original seller
   string) so ``dataset_loaders.py`` can look up splits without re-running
   NER at training time.

Usage
-----
    python preprocess_seller_split.py

The script requires:
    pip install spacy datasets
    python -m spacy download en_core_web_sm

If the HuggingFace dataset download is unavailable (e.g. in a network-
restricted sandbox), the script falls back to a built-in list of
representative seller strings and documents the fallback in the cache
metadata.

Environment variables (optional, for higher-accuracy LLM fallback):
    OPENAI_API_KEY    — use OpenAI GPT-4o for hard cases
    ANTHROPIC_API_KEY — use Anthropic Claude for hard cases
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

__all__ = ["build_cache", "main"]

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
CACHE_PATH = REPO_ROOT / "seller_split_cache.json"

# ---------------------------------------------------------------------------
# Compiled heuristics (fallback when NER confidence is low)
# ---------------------------------------------------------------------------

_STREET_NUMBER_RE = re.compile(r"\b\d+\s+[A-Za-z]")
_PO_BOX_RE = re.compile(r"\bP\.?\s*O\.?\s*Box\b", re.IGNORECASE)
_STATE_ZIP_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")
_STREET_SUFFIX_RE = re.compile(
    r"\b(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|"
    r"Way|Lane|Ln|Court|Ct|Place|Pl|Terrace|Ter|Circle|Cir|"
    r"Highway|Hwy|Parkway|Pkwy|Trail|Trl|Run|Loop|Row)\b",
    re.IGNORECASE,
)


def _heuristic_split(seller: str) -> tuple[str, str]:
    """Split seller string into (company, address) using regex heuristics.

    Looks for the earliest address indicator (street number, P.O. Box,
    street suffix, or state+ZIP) that appears *after* the first character.
    Returns (full_string, "") when no address pattern is found.
    """
    seller = seller.strip()
    if not seller:
        return ("", "")

    candidates: list[int] = []
    for pattern in (_PO_BOX_RE, _STREET_NUMBER_RE, _STREET_SUFFIX_RE, _STATE_ZIP_RE):
        m = pattern.search(seller)
        if m and m.start() > 0:
            candidates.append(m.start())

    if not candidates:
        return (seller, "")

    split_at = min(candidates)
    company = seller[:split_at].strip().rstrip(",").strip()
    address = seller[split_at:].strip()
    return (company, address)


# ---------------------------------------------------------------------------
# spaCy-based split
# ---------------------------------------------------------------------------


def _spacy_split(seller: str, nlp) -> tuple[str, str] | None:
    """Use spaCy NER to split seller into (company, address).

    Returns None when NER cannot provide a confident split (no ORG/PERSON
    entities found), so callers can fall back to heuristics.

    Strategy:
    - Find the last ORG or PERSON entity span end as the company boundary.
    - Everything from that boundary onward is the address portion.
    - If no address-type entity (GPE/LOC/FAC) is found after the company
      boundary, fall back to heuristic on the remaining substring.
    """
    doc = nlp(seller)

    # Find the rightmost end of any ORG or PERSON entity.
    company_end: int | None = None
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PERSON") and (company_end is None or ent.end_char > company_end):
            company_end = ent.end_char

    if company_end is None:
        # No clear company entity — cannot split with confidence.
        return None

    company_candidate = seller[:company_end].strip().rstrip(",").strip()
    remainder = seller[company_end:].strip().lstrip(",").strip()

    if not remainder:
        # Entire string is company; check heuristics on full string.
        _, addr = _heuristic_split(seller)
        if addr:
            split_at = seller.find(addr)
            company_candidate = seller[:split_at].strip().rstrip(",").strip()
            return (company_candidate, addr)
        return (seller.strip(), "")

    # Verify the remainder looks like an address via heuristics.
    _, addr_check = _heuristic_split(remainder)
    if addr_check or _STATE_ZIP_RE.search(remainder):
        return (company_candidate, remainder)

    # Remainder doesn't look like an address; use heuristic on full string.
    heur = _heuristic_split(seller)
    return heur


# ---------------------------------------------------------------------------
# LLM fallback (optional)
# ---------------------------------------------------------------------------


def _llm_split(seller: str) -> tuple[str, str] | None:
    """Try OpenAI or Anthropic API to split a hard seller string.

    Returns None when no API key is available or the API call fails.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    prompt = (
        f"Split this seller string into company name and street address.\n"
        f'Input: "{seller}"\n'
        f'Output JSON only (no commentary): {{"company": "...", "address": "..."}}'
    )

    if openai_key:
        try:
            import openai  # type: ignore

            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            obj = json.loads(text)
            return (str(obj.get("company", "")), str(obj.get("address", "")))
        except Exception as exc:
            log.debug("OpenAI fallback failed for %r: %s", seller, exc)

    if anthropic_key:
        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            obj = json.loads(text)
            return (str(obj.get("company", "")), str(obj.get("address", "")))
        except Exception as exc:
            log.debug("Anthropic fallback failed for %r: %s", seller, exc)

    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_seller_strings() -> tuple[list[str], str]:
    """Load all seller strings from the HuggingFace dataset.

    Returns (seller_strings, source_description).
    Falls back to a built-in representative list if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset  # type: ignore

        log.info("Loading katanaml-org/invoices-donut-data-v1 from HuggingFace …")
        ds = load_dataset("katanaml-org/invoices-donut-data-v1")
        sellers: list[str] = []
        for split_name, split_ds in ds.items():
            if split_name == "test":
                continue
            for item in split_ds:
                gt_str = item.get("ground_truth", "{}")
                try:
                    obj = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
                    gt_parse = obj.get("gt_parse", obj)
                    header = gt_parse.get("header", {})
                    if isinstance(header, dict):
                        seller = str(header.get("seller", "")).strip()
                        if seller:
                            sellers.append(seller)
                except (json.JSONDecodeError, AttributeError):
                    pass
        log.info("Loaded %d seller strings from HuggingFace.", len(sellers))
        return sellers, "katanaml-org/invoices-donut-data-v1 (HuggingFace)"

    except Exception as exc:
        log.warning(
            "HuggingFace dataset unavailable (%s). "
            "Falling back to representative built-in examples.",
            exc,
        )

    # Built-in representative examples drawn from the dataset's documented
    # distribution (used when HuggingFace is unreachable in the sandbox).
    builtin = [
        "Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228",
        "Smith & Associates LLC 1200 Oak Avenue Suite 400 Chicago, IL 60601",
        "Johnson Industrial Corp P.O. Box 4521 Houston, TX 77001",
        "Williams Group 888 Maple Street Portland, OR 97201",
        "Brown Consulting Ltd 42 Elm Road Austin, TX 78701",
        "Davis & Partners 3500 North Blvd Denver, CO 80201",
        "Martinez Enterprises 500 Commerce Drive Miami, FL 33101",
        "Anderson Technology Inc 720 Tech Parkway Seattle, WA 98101",
        "Thomas Manufacturing 2100 Industrial Way Detroit, MI 48201",
        "Jackson & Sons 99 River Lane Nashville, TN 37201",
        "White Solutions 1 Corporate Court Boston, MA 02101",
        "Harris Financial Group 555 Wall Street New York, NY 10005",
        "Clark Logistics 3000 Airport Blvd Los Angeles, CA 90001",
        "Lewis Brothers 601 Oak Terrace Atlanta, GA 30301",
        "Robinson Retail Inc 250 Mall Circle Phoenix, AZ 85001",
        "Walker Industries 10 Harbor Drive San Francisco, CA 94101",
        "Hall Healthcare 800 Medical Parkway Dallas, TX 75201",
        "Allen & Baker 404 Research Blvd Philadelphia, PA 19101",
        "Young Law Group 77 Justice Lane Minneapolis, MN 55401",
        "Hernandez Construction 3100 Highway 1 San Antonio, TX 78201",
    ]
    return builtin, "built-in representative examples (HuggingFace unavailable)"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def build_cache(
    sellers: list[str],
    nlp,
    use_llm: bool = True,
) -> dict[str, dict[str, str]]:
    """Process a list of seller strings and return the split cache dict."""
    cache: dict[str, dict[str, str]] = {}
    n_spacy = n_heuristic = n_llm = 0

    unique_sellers = list(dict.fromkeys(sellers))  # deduplicate, preserve order
    log.info("Processing %d unique seller strings …", len(unique_sellers))

    for seller in unique_sellers:
        result = _spacy_split(seller, nlp)
        method = "spacy"

        if result is None and use_llm:
            result = _llm_split(seller)
            if result is not None:
                method = "llm"

        if result is None:
            result = _heuristic_split(seller)
            method = "heuristic"

        company, address = result
        cache[seller] = {"company": company, "address": address}

        if method == "spacy":
            n_spacy += 1
        elif method == "llm":
            n_llm += 1
        else:
            n_heuristic += 1

    log.info(
        "Split breakdown — spaCy: %d, heuristic: %d, LLM: %d",
        n_spacy,
        n_heuristic,
        n_llm,
    )
    return cache


def main() -> int:
    # ── Load spaCy model ──────────────────────────────────────────────────
    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load("en_core_web_sm")
            log.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            log.warning("en_core_web_sm not found; downloading …")
            import subprocess

            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                stdout=subprocess.DEVNULL,
            )
            nlp = spacy.load("en_core_web_sm")
    except ImportError:
        log.error("spaCy is not installed. Run: pip install spacy")
        return 1

    # ── Load seller strings ───────────────────────────────────────────────
    sellers, source = _load_seller_strings()

    # ── Build cache ───────────────────────────────────────────────────────
    cache = build_cache(sellers, nlp)

    # ── Write output ──────────────────────────────────────────────────────
    output: dict = {
        "_metadata": {
            "source": source,
            "num_entries": len(cache),
            "note": (
                "Generated by preprocess_seller_split.py using spaCy NER "
                "(en_core_web_sm) with heuristic fallback. "
                "Re-run the script with internet access to regenerate from "
                "the full HuggingFace dataset."
            ),
        }
    }
    output.update(cache)

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info("Wrote %d entries to %s", len(cache), CACHE_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
