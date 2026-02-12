# Receipt Image Processing Guide

## Overview

This guide explains how to use the receipt image processing functionality in `kaggle_receipt_ocr.py`, a single-file OCR receipt processing system designed for Kaggle and easy deployment.

## Features

The `process_image` function and associated `process` command provide:

- **Automatic Receipt Image Processing**: Load and preprocess receipt images
- **OCR Text Extraction**: Extract text using EasyOCR or Tesseract
- **Smart Data Parsing**: Extract structured data (store name, date, total, tax)
- **CSV Export**: Save results to timestamped CSV files
- **Flexible Output**: Configure output directory

## Quick Start

### Command Line Usage

#### Process a Single Receipt Image

```bash
python kaggle_receipt_ocr.py process receipt.jpg
```

This will:
1. Load and preprocess the image
2. Extract text using OCR
3. Parse receipt data (store, date, total, tax)
4. Save results to `output/receipt_data_<timestamp>.csv`

#### Process with Custom Output Directory

```bash
python kaggle_receipt_ocr.py process receipt.jpg my_output_folder
```

Results will be saved to `my_output_folder/receipt_data_<timestamp>.csv`

### Programmatic Usage

#### Basic Example

```python
from kaggle_receipt_ocr import process_image

# Process a receipt image
receipt = process_image("receipt.jpg")

# Access extracted data
print(f"Store: {receipt.store_name}")
print(f"Date: {receipt.date}")
print(f"Total: ${receipt.total:.2f}" if receipt.total else "Total: N/A")
print(f"Processing time: {receipt.processing_time:.2f}s")
```

#### Advanced Example with Custom Output

```python
from kaggle_receipt_ocr import process_image, ReceiptProcessor
from pathlib import Path

# Process with custom output directory
output_dir = "my_receipts"
receipt = process_image("receipt.jpg", output_dir)

# Or use the ReceiptProcessor class directly
processor = ReceiptProcessor()
receipt = processor.process_receipt("receipt.jpg")

# Save results manually
custom_path = Path("custom_results.csv")
processor.save_results(receipt, custom_path)
```

## Processing Workflow

The receipt image processing follows these steps:

### 1. Image Loading
- Loads the image file using PIL/Pillow
- Validates the image can be opened
- Handles various image formats (JPG, PNG, etc.)

### 2. Image Preprocessing
- Converts image to grayscale
- Enhances contrast (1.5x enhancement)
- Sharpens the image (1.5x enhancement)
- Optimizes image for OCR accuracy

### 3. OCR Text Extraction
- **Primary Engine**: EasyOCR (if available)
  - Uses confidence threshold (default: 0.3)
  - Runs on CPU (GPU disabled for Kaggle compatibility)
  - Returns high-quality text extraction
- **Fallback Engine**: Tesseract OCR (if EasyOCR unavailable)
  - Provides reliable backup text extraction
  - Works on most systems

### 4. Data Parsing
Extracts structured information from OCR text:

- **Store Name**: 
  - Matches against known stores list
  - Looks for capitalized text at top of receipt
  - Returns first match found

- **Date**: 
  - Multiple format support (YYYY-MM-DD, MM/DD/YYYY, etc.)
  - Regex pattern matching
  - Standardized output format

- **Total Amount**: 
  - Searches for "TOTAL", "AMOUNT DUE", "BALANCE" keywords
  - Extracts monetary values
  - Fallback to largest dollar amount if keywords not found

- **Tax Amount**: 
  - Searches for "TAX", "VAT" keywords
  - Extracts associated monetary value

- **Receipt Type Classification**:
  - GROCERY: Walmart, Target, Kroger, Safeway, etc.
  - RESTAURANT: Starbucks, McDonald's, Subway, etc.
  - RETAIL: Best Buy, general stores, etc.
  - GAS_STATION: Gas stations
  - UNKNOWN: Unclassified receipts

### 5. Results Export
- Creates timestamped CSV file
- Includes all extracted data fields
- UTF-8 encoding for international characters
- Creates output directory if it doesn't exist

## Output Format

### CSV Structure

The output CSV contains the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `store_name` | String | Name of the store | "WALMART" |
| `date` | String | Receipt date | "01/15/2024" |
| `time` | String | Receipt time (if available) | "14:30:25" |
| `total` | Float | Total amount | 45.32 |
| `tax` | Float | Tax amount | 3.63 |
| `subtotal` | Float | Subtotal (if available) | 41.69 |
| `receipt_type` | String | Type of receipt | "grocery" |
| `confidence` | Float | Parsing confidence (0-1) | 0.85 |
| `ocr_raw_text` | String | Raw OCR output | "WALMART\nDate: ..." |
| `processing_time` | Float | Processing time in seconds | 2.45 |

### ReceiptData Object

When using the Python API, `process_image` returns a `ReceiptData` object:

```python
receipt = process_image("receipt.jpg")

# Access attributes
receipt.store_name     # Store name string
receipt.date          # Date string
receipt.time          # Time string (optional)
receipt.total         # Total amount (float)
receipt.tax           # Tax amount (float)
receipt.subtotal      # Subtotal (float, optional)
receipt.receipt_type  # ReceiptType enum
receipt.confidence    # Confidence score (float)
receipt.ocr_raw_text  # Full OCR text
receipt.processing_time  # Processing duration

# Convert to dictionary
data_dict = receipt.to_dict()
```

## Requirements

### System Dependencies

- Python 3.7+
- PIL/Pillow (image processing)

### Optional OCR Engines

For full functionality, install at least one OCR engine:

**EasyOCR** (Recommended):
```bash
pip install easyocr
```

**Tesseract** (Fallback):
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

# Python wrapper
pip install pytesseract
```

### Python Dependencies

```bash
pip install numpy matplotlib pillow
```

## Configuration

### Customizing Known Stores

Edit the `KNOWN_STORES` list in the `Config` class:

```python
KNOWN_STORES = [
    'WALMART', 'TARGET', 'COSTCO',
    'YOUR_CUSTOM_STORE',  # Add your stores here
    # ...
]
```

### Adjusting OCR Confidence

Modify the confidence threshold (0.0-1.0):

```python
OCR_CONFIDENCE_THRESHOLD = 0.3  # Lower = more results, Higher = more accurate
```

### Changing Image Size Limits

Adjust maximum image size for memory constraints:

```python
IMAGE_MAX_SIZE = (2000, 2000)  # (width, height) in pixels
```

## Error Handling

The system handles common errors gracefully:

### No OCR Engine Available
```
WARNING: EasyOCR not available
WARNING: Tesseract not available
ERROR: No OCR engine available!
```

**Solution**: Install at least one OCR engine (EasyOCR or Tesseract)

### Image Loading Failure
```
ERROR: Failed to load image
```

**Solution**: 
- Check file path is correct
- Verify image format is supported (JPG, PNG)
- Ensure Pillow is installed

### Empty Text Extraction
```
Extracted 0 characters
```

**Solution**:
- Check image quality
- Ensure receipt text is readable
- Try preprocessing the image manually
- Install a more robust OCR engine

## Examples

### Example 1: Batch Processing

```python
from kaggle_receipt_ocr import process_image
from pathlib import Path
import csv

# Process multiple receipts
receipt_dir = Path("receipts")
results = []

for image_path in receipt_dir.glob("*.jpg"):
    receipt = process_image(str(image_path), "batch_output")
    results.append(receipt.to_dict())

# Combine into single CSV
with open("all_receipts.csv", 'w', newline='') as f:
    if results:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

print(f"Processed {len(results)} receipts")
```

### Example 2: Extract Specific Data

```python
from kaggle_receipt_ocr import process_image

# Process receipt
receipt = process_image("receipt.jpg")

# Extract only what you need
if receipt.total and receipt.store_name:
    print(f"Spent ${receipt.total:.2f} at {receipt.store_name}")
else:
    print("Could not extract complete data")
```

### Example 3: Custom Processing Pipeline

```python
from kaggle_receipt_ocr import ReceiptProcessor, ImageProcessor
from PIL import Image

# Create processor
processor = ReceiptProcessor()

# Load and preprocess image manually
image = Image.open("receipt.jpg")
image = processor.image_processor.preprocess_for_ocr(image)

# Extract text
text = processor.ocr_engine.extract_text(image)

# Parse receipt
receipt = processor.parser.parse_receipt(text)

# Custom output
print(f"Extracted {len(text)} characters")
print(f"Found store: {receipt.store_name or 'Unknown'}")
```

## Performance Tips

1. **Image Quality**: Higher quality images produce better results
2. **Image Size**: Resize very large images to reduce processing time
3. **OCR Engine**: EasyOCR generally provides better results than Tesseract
4. **Preprocessing**: The built-in preprocessing works well for most receipts
5. **Batch Processing**: Process multiple receipts in parallel for better throughput

## Troubleshooting

### Poor Extraction Results

If extraction quality is poor:

1. **Check image quality**: Ensure text is readable
2. **Try manual preprocessing**: Adjust contrast, brightness
3. **Use different OCR engine**: Try EasyOCR if using Tesseract, or vice versa
4. **Adjust confidence threshold**: Lower for more results, raise for accuracy

### Memory Issues

If running out of memory:

1. **Reduce IMAGE_MAX_SIZE**: Lower the maximum image dimensions
2. **Process one at a time**: Avoid parallel processing
3. **Close images**: Explicitly close PIL images after processing

### Slow Processing

If processing is slow:

1. **Enable GPU**: Modify EasyOCR initialization to use GPU
2. **Reduce image size**: Resize images before processing
3. **Use faster engine**: Tesseract may be faster on some systems
4. **Limit preprocessing**: Skip enhancement steps if not needed

## Support

For issues or questions:
1. Check this guide for common solutions
2. Review the code documentation in `kaggle_receipt_ocr.py`
3. Open a GitHub issue with example images and error messages

## Version Information

- **File**: `kaggle_receipt_ocr.py`
- **Version**: 3.0.0
- **Single File**: 577 lines, ~20KB
- **Last Updated**: 2026-02-12

---

**Made for Kaggle and easy deployment! 🎉**
