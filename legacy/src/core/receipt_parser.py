"""Receipt parsing and information extraction."""

import re
from typing import List, Dict, Optional
from datetime import datetime

from ..utils.models import LineItem


class ReceiptParser:
    """Extract structured information from OCR text."""
    
    def __init__(self, config=None):
        """Initialize parser with config."""
        if config is None:
            from ..utils.config import Config
            self.config = Config
        else:
            self.config = config
    
    def extract_store_name(self, text: str, ocr_boxes: List[Dict] = None) -> Optional[str]:
        """Extract store name from receipt text."""
        lines = text.split('\n')
        
        # Check first few lines for known stores
        for line in lines[:10]:
            line_upper = line.upper().strip()
            for store in self.config.KNOWN_STORES:
                if store in line_upper:
                    return store
        
        # Check OCR boxes if available (usually store name is at top with larger font)
        if ocr_boxes:
            # Sort by y-position
            sorted_boxes = sorted(ocr_boxes, key=lambda x: x['position']['y'])
            for box in sorted_boxes[:5]:
                text_upper = box['text'].upper()
                for store in self.config.KNOWN_STORES:
                    if store in text_upper:
                        return store
        
        # Try to find capitalized words at the top
        for line in lines[:5]:
            words = [w for w in line.strip().split() if w]  # Filter empty strings
            if words and len(words) <= 3:
                if all(word[0].isupper() for word in words):
                    return ' '.join(words)
        
        return None
    
    def extract_date(self, text: str) -> Optional[str]:
        """Extract date from receipt text."""
        for pattern in self.config.DATE_FORMATS:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(0)
                # Try to parse and standardize
                try:
                    # Try various parsing methods
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d.%m.%Y', 
                               '%m/%d/%y', '%Y%m%d']:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            return dt.strftime('%Y-%m-%d')
                        except:
                            continue
                    return date_str
                except:
                    return date_str
        return None
    
    def extract_time(self, text: str) -> Optional[str]:
        """Extract time from receipt text."""
        for pattern in self.config.TIME_FORMATS:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def extract_total(self, text: str) -> Optional[float]:
        """Extract total amount from receipt text."""
        for pattern in self.config.TOTAL_PATTERNS:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        
        # Fallback: find largest monetary amount
        amounts = re.findall(r'\$?(\d+\.\d{2})', text)
        if amounts:
            try:
                return max(float(amt) for amt in amounts)
            except:
                pass
        
        return None
    
    def extract_tax(self, text: str) -> Optional[float]:
        """Extract tax amount from receipt text."""
        for pattern in self.config.TAX_PATTERNS:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        return None
    
    def extract_subtotal(self, text: str) -> Optional[float]:
        """Extract subtotal from receipt text."""
        patterns = [
            r'(?i)subtotal\s*:?\s*\$?(\d+\.?\d*)',
            r'(?i)sub\s*total\s*:?\s*\$?(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        return None
    
    def extract_currency(self, text: str) -> str:
        """Detect currency from receipt text."""
        for currency, pattern in self.config.CURRENCIES.items():
            if re.search(pattern, text):
                return currency
        return "USD"  # Default
    
    def extract_payment_method(self, text: str) -> Optional[str]:
        """Extract payment method."""
        text_upper = text.upper()
        for method in self.config.PAYMENT_METHODS:
            if method in text_upper:
                return method
        return None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number."""
        patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def extract_address(self, text: str) -> Optional[str]:
        """Extract address (simplified)."""
        # Look for patterns like "123 Main St"
        pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
        return None
    
    def extract_receipt_number(self, text: str) -> Optional[str]:
        """Extract receipt number."""
        patterns = [
            r'(?i)receipt\s*#?\s*:?\s*(\w+)',
            r'(?i)trans(?:action)?\s*#?\s*:?\s*(\w+)',
            r'(?i)order\s*#?\s*:?\s*(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def extract_line_items(self, text: str) -> List[LineItem]:
        """Extract line items from receipt."""
        items = []
        lines = text.split('\n')
        
        # Simple pattern matching for items
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse as line item
            item = LineItem.from_text(line)
            if item:
                items.append(item)
        
        return items
    
    def parse_receipt(self, text: str, ocr_boxes: List[Dict] = None) -> Dict:
        """Parse receipt and extract all information."""
        result = {
            'store_name': self.extract_store_name(text, ocr_boxes),
            'date': self.extract_date(text),
            'time': self.extract_time(text),
            'total': self.extract_total(text),
            'tax': self.extract_tax(text),
            'subtotal': self.extract_subtotal(text),
            'currency': self.extract_currency(text),
            'payment_method': self.extract_payment_method(text),
            'phone': self.extract_phone(text),
            'address': self.extract_address(text),
            'receipt_number': self.extract_receipt_number(text),
            'items': self.extract_line_items(text)
        }
        return result
