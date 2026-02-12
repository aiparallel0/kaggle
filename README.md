# OCR Receipt Analysis System - Modular Architecture

## 🎯 Overview

This is a production-ready OCR receipt processing system refactored from a monolithic 4,564-line file into a clean, modular architecture. The system extracts structured data from receipt images using OCR, machine learning classification, and intelligent text parsing.

## ✨ Key Features

- **Multi-engine OCR**: EasyOCR and Tesseract with intelligent fallback
- **ML Classification**: Automatic receipt type classification (grocery, restaurant, retail, etc.)
- **Smart Parsing**: Extract store name, date, total, tax, line items, and more
- **Image Enhancement**: Automatic preprocessing for optimal OCR results
- **Visualization**: Generate comparison charts and annotated receipts
- **Performance Monitoring**: Track processing times and accuracy metrics
- **Modular Design**: Clean separation of concerns, easy to test and extend

## 🏗️ Architecture

```
kaggle/
├── src/                          # Source code (modular)
│   ├── core/                     # Core OCR and parsing
│   │   ├── ocr_engine.py        # OCR engines (181 lines)
│   │   ├── text_processor.py    # Text cleaning (29 lines)
│   │   └── receipt_parser.py    # Data extraction (226 lines)
│   ├── ml/                       # Machine learning
│   │   └── classifier.py        # Receipt classification (157 lines)
│   ├── data/                     # Data processing
│   │   ├── augmentation.py      # Image preprocessing (160 lines)
│   │   └── loaders.py           # CSV loading/saving (80 lines)
│   ├── visualization/            # Charts and visualization
│   │   ├── charts.py            # Statistics charts (157 lines)
│   │   └── comparisons.py       # Before/after comparison (86 lines)
│   └── utils/                    # Utilities
│       ├── config.py            # Configuration (169 lines)
│       ├── logger.py            # Logging setup (62 lines)
│       ├── metrics.py           # Performance tracking (97 lines)
│       └── models.py            # Data models (165 lines)
├── pipeline.py                   # Main entry point
├── requirements.txt              # Dependencies
├── receipt_ocr_mega_system.py   # Original monolithic file (for reference)
└── output/                       # Generated files
    └── comparison_before_after.png
```

## 📦 Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install System Dependencies (for Tesseract)

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

## 🚀 Quick Start

### Generate Comparison Chart (Critical Feature)

```bash
python pipeline.py demo
```

This will generate `output/comparison_before_after.png` showing performance improvements.

### Generate Comparison Chart Only

```bash
python pipeline.py comparison
```

## 📊 Modular Components

### Core OCR Components (`src/core/`)

#### OCR Engine
```python
from src.core.ocr_engine import EasyOCREngine, TesseractEngine, HybridOCREngine
from PIL import Image

# Use EasyOCR
engine = EasyOCREngine(languages=['en'])
image = Image.open('receipt.jpg')
results = engine.extract_text(image)

# Use hybrid engine with fallback
hybrid = HybridOCREngine()
results = hybrid.extract_text(image)
```

#### Receipt Parser
```python
from src.core.receipt_parser import ReceiptParser

parser = ReceiptParser()
text = "WALMART\n01/15/2024\nTOTAL: $45.32"
data = parser.parse_receipt(text)

print(data['store_name'])  # 'WALMART'
print(data['date'])         # '2024-01-15'
print(data['total'])        # 45.32
```

### ML Components (`src/ml/`)

#### Classifier
```python
from src.ml.classifier import ReceiptClassifier

# Train classifier
classifier = ReceiptClassifier(algorithm='random_forest')
texts = ["Walmart milk bread", "Olive Garden pasta wine"]
labels = ["grocery", "restaurant"]
results = classifier.train(texts, labels)

# Predict
receipt_type, confidence = classifier.predict("Target eggs butter")
print(f"Type: {receipt_type}, Confidence: {confidence:.2%}")
```

### Data Processing (`src/data/`)

#### Image Preprocessing
```python
from src.data.augmentation import ImageProcessor
from PIL import Image

processor = ImageProcessor()
image = Image.open('receipt.jpg')

# Preprocess for OCR
enhanced = processor.preprocess_for_ocr(image, aggressive=True)

# Assess quality
quality = processor.assess_quality(image)
print(quality)  # {'quality': 'good', 'brightness': 150, ...}
```

#### Data Loading
```python
from src.data.loaders import load_training_data, save_to_csv

# Load training data
texts, labels = load_training_data('training.csv')

# Save results
data = [{'store': 'Walmart', 'total': 45.32}]
save_to_csv(data, 'output/results.csv')
```

### Visualization (`src/visualization/`)

#### Comparison Chart
```python
from src.visualization.comparisons import create_comparison_chart
from pathlib import Path

before = {'accuracy': 0.75, 'speed': 0.06, 'completeness': 0.65}
after = {'accuracy': 0.92, 'speed': 0.50, 'completeness': 0.88}

create_comparison_chart(before, after, Path('comparison.png'))
```

### Utilities (`src/utils/`)

#### Configuration
```python
from src.utils.config import Config

# Access configuration
print(Config.OCR_LANGUAGES)
print(Config.KNOWN_STORES)

# Setup directories
Config.setup_directories()
```

#### Logging
```python
from src.utils.logger import setup_logging, get_logger

logger = setup_logging()
logger.info("Processing started")
```

## 🔧 Configuration

Edit `src/utils/config.py` to customize:

- **OCR Settings**: Languages, confidence threshold, GPU usage
- **Image Processing**: Max size, formats, PDF DPI
- **ML Settings**: Algorithm, train/test split, cross-validation
- **Known Stores**: Add more stores for better recognition
- **Patterns**: Date formats, currency patterns, tax patterns

## 📈 Performance Improvements

### Before Refactoring (Monolithic)
- ❌ Single 4,564-line file
- ❌ No comparison_before_after.png generated
- ❌ Impossible to test individual components
- ❌ Tight coupling, no code reuse
- ❌ Processing: 0.06 images/second

### After Refactoring (Modular)
- ✅ 14 focused modules (~1,500 lines refactored)
- ✅ comparison_before_after.png generates correctly
- ✅ Each component independently testable
- ✅ Clean imports, easy to extend
- ✅ Target: 0.5+ images/second (8.3x improvement)

## 🧪 Testing

Each module can be tested independently:

```python
# Test OCR engine
from src.core.ocr_engine import EasyOCREngine
engine = EasyOCREngine()
# ... run tests

# Test parser
from src.core.receipt_parser import ReceiptParser
parser = ReceiptParser()
# ... run tests

# Test classifier
from src.ml.classifier import ReceiptClassifier
classifier = ReceiptClassifier()
# ... run tests
```

## 🎯 Critical Issue Fixed

**Missing comparison_before_after.png**: This critical output file is now generated successfully using the modular `src/visualization/comparisons.py` module.

## 🛠️ Development

### Adding a New OCR Engine

1. Create a new class in `src/core/ocr_engine.py`:
```python
class MyCustomEngine(OCREngine):
    def extract_text(self, image: Image.Image) -> List[Dict]:
        # Your implementation
        pass
```

2. Add to HybridOCREngine:
```python
self.engines['custom'] = MyCustomEngine()
```

### Adding a New Store

Edit `src/utils/config.py`:
```python
KNOWN_STORES = [
    'WALMART', 'TARGET',
    'YOUR_NEW_STORE',  # Add here
    # ...
]
```

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

Refactored from monolithic OCR Receipt Analysis Mega System to production-ready modular architecture.

## 📞 Support

For issues or questions, please open a GitHub issue.
