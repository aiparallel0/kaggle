# Kaggle OCR Receipt Analysis - Single File Version

## 🎯 Perfect for Kaggle!

This is a **single-file** OCR receipt processing system designed specifically for Kaggle notebooks. Just **577 lines** in one file - easy to upload and run!

## ✨ Features

- ✅ **Single File** - No complex imports or module structure
- ✅ **OCR Text Extraction** - EasyOCR with Tesseract fallback
- ✅ **Smart Receipt Parsing** - Extracts store, date, total, tax
- ✅ **Comparison Charts** - Before/after visualization
- ✅ **Image Preprocessing** - Automatic enhancement for better OCR
- ✅ **Kaggle-Ready** - Works out of the box on Kaggle

## 🚀 Quick Start on Kaggle

### Step 1: Upload File
Upload `kaggle_receipt_ocr.py` to your Kaggle notebook

### Step 2: Install Dependencies
```python
!pip install numpy matplotlib pillow easyocr
```

### Step 3: Run Demo
```python
!python kaggle_receipt_ocr.py demo
```

## 📝 Usage Examples

### Run Demo (Test Everything)
```python
!python kaggle_receipt_ocr.py demo
```

### Process a Receipt Image
```python
!python kaggle_receipt_ocr.py process receipt.jpg
```

### Generate Comparison Chart
```python
!python kaggle_receipt_ocr.py comparison
```

## 📊 What You Get

### Demo Output
- ✅ Comparison chart: `output/comparison_before_after.png`
- ✅ Receipt parsing test with sample data
- ✅ Performance metrics

### Processing Output
- CSV file with extracted data:
  - Store name
  - Date
  - Total amount
  - Tax
  - Receipt type
  - Processing time

## 🎓 Using in Your Kaggle Notebook

```python
# Import the module
import sys
sys.path.append('/kaggle/input')  # Adjust path as needed

# Run directly
!python /kaggle/input/kaggle_receipt_ocr.py demo

# Or use as a library
from kaggle_receipt_ocr import ReceiptProcessor

processor = ReceiptProcessor()
receipt = processor.process_receipt('receipt.jpg')

print(f"Store: {receipt.store_name}")
print(f"Total: ${receipt.total:.2f}")
```

## 📦 Dependencies

The file uses standard Python libraries plus:
- `numpy` - For numerical operations
- `matplotlib` - For charts
- `Pillow` - For image processing
- `easyocr` (optional) - Primary OCR engine
- `pytesseract` (optional) - Fallback OCR

## 🎯 Perfect For

- ✅ Kaggle competitions with receipt data
- ✅ Quick OCR experiments
- ✅ Learning OCR and receipt processing
- ✅ Prototyping receipt analysis pipelines
- ✅ Sharing code easily (just one file!)

## 📏 File Stats

- **Lines**: 577 lines
- **Size**: 20KB
- **Functions**: 20+ functions
- **Classes**: 5 core classes
- **No external module dependencies** (all in one file!)

## 🔧 Customization

Edit these sections in the file:

### Add More Stores
```python
KNOWN_STORES = [
    'WALMART', 'TARGET', 'COSTCO',
    'YOUR_STORE_HERE',  # Add your stores
]
```

### Adjust OCR Confidence
```python
OCR_CONFIDENCE_THRESHOLD = 0.3  # Lower = more results, higher = more accurate
```

### Change Image Size
```python
IMAGE_MAX_SIZE = (2000, 2000)  # Adjust for memory constraints
```

## 🎉 Why Single File?

### Benefits
1. **Easy Upload** - Just drag and drop to Kaggle
2. **No Import Issues** - Everything in one place
3. **Easy to Share** - Send one file, not a folder
4. **Kaggle-Friendly** - Works with Kaggle's file system
5. **Self-Contained** - No complex dependencies

### Comparison with Modular Version

| Feature | Single File | Modular (src/) |
|---------|-------------|----------------|
| Lines of Code | 577 | 1,569 (14 files) |
| Kaggle Upload | ✅ Easy | ❌ Complex |
| Maintainability | ⚠️ Good | ✅ Excellent |
| Team Development | ⚠️ Harder | ✅ Easy |
| Quick Experiments | ✅ Perfect | ⚠️ Overkill |

## 📖 Documentation

The file is well-documented with:
- Clear section headers
- Docstrings for all functions
- Inline comments for complex logic
- Usage examples at the top

## 🆘 Troubleshooting

### "No module named 'numpy'"
```python
!pip install numpy matplotlib
```

### "EasyOCR not available"
```python
!pip install easyocr
# Or use Tesseract: !apt-get install tesseract-ocr
```

### "Failed to load image"
```python
!pip install Pillow
```

## 🎊 Success!

You now have a powerful OCR receipt processing system in **just one file** that's perfect for Kaggle!

---

**Made for Kaggle with ❤️**
