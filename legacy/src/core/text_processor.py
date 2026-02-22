"""Text processing and extraction utilities for receipt OCR."""

import re
from typing import List, Optional
from datetime import datetime


class TextProcessor:
    """Process and clean extracted OCR text."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep common punctuation
        text = re.sub(r'[^\w\s$.,:\-/()&]', '', text)
        return text.strip()
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        return text.upper().strip()
    
    @staticmethod
    def extract_lines(text: str) -> List[str]:
        """Extract lines from text, removing empty lines."""
        lines = [line.strip() for line in text.split('\n')]
        return [line for line in lines if line]
