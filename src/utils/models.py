"""Data models for OCR receipt analysis system."""

import re
import json
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict


class ReceiptType(str, Enum):
    """Types of receipts."""
    GROCERY = "grocery"
    RESTAURANT = "restaurant"
    RETAIL = "retail"
    GAS_STATION = "gas_station"
    PHARMACY = "pharmacy"
    ONLINE = "online"
    SERVICE = "service"
    ENTERTAINMENT = "entertainment"
    TRANSPORT = "transport"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class LineItem:
    """Represents a single line item on a receipt."""
    description: str
    quantity: float = 1.0
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    category: Optional[str] = None
    tax_amount: Optional[float] = None
    discount: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_text(cls, text: str) -> Optional['LineItem']:
        """Extract line item from text."""
        # Pattern: Description Price or Description Quantity x UnitPrice = Total
        patterns = [
            r'(.+?)\s+(\d+\.?\d*)\s*x\s*\$?(\d+\.?\d*)\s*=?\s*\$?(\d+\.?\d*)',
            r'(.+?)\s+\$?(\d+\.?\d*)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.strip())
            if match:
                if len(match.groups()) == 4:
                    desc, qty, unit, total = match.groups()
                    return cls(
                        description=desc.strip(),
                        quantity=float(qty),
                        unit_price=float(unit),
                        total_price=float(total)
                    )
                elif len(match.groups()) == 2:
                    desc, price = match.groups()
                    return cls(
                        description=desc.strip(),
                        total_price=float(price)
                    )
        return None


@dataclass
class ReceiptData:
    """Complete receipt data structure."""
    
    # Basic information
    receipt_id: str = ""
    store_name: Optional[str] = None
    store_address: Optional[str] = None
    store_phone: Optional[str] = None
    
    # Transaction details
    date: Optional[str] = None
    time: Optional[str] = None
    receipt_number: Optional[str] = None
    transaction_id: Optional[str] = None
    cashier: Optional[str] = None
    
    # Financial information
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    tip: Optional[float] = None
    discount: Optional[float] = None
    total: Optional[float] = None
    amount_paid: Optional[float] = None
    change: Optional[float] = None
    
    # Payment information
    payment_method: Optional[str] = None
    card_last_four: Optional[str] = None
    
    # Line items
    items: List[LineItem] = field(default_factory=list)
    
    # Classification
    receipt_type: ReceiptType = ReceiptType.UNKNOWN
    confidence: float = 0.0
    
    # Currency
    currency: str = "USD"
    
    # Metadata
    ocr_raw_text: str = ""
    processing_time: float = 0.0
    image_quality: str = "unknown"
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['items'] = [item.to_dict() if isinstance(item, LineItem) else item 
                        for item in self.items]
        data['created_at'] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReceiptData':
        """Create from dictionary."""
        if 'items' in data:
            data['items'] = [LineItem(**item) if isinstance(item, dict) else item 
                           for item in data['items']]
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
