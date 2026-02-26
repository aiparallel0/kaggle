# Receipt OCR Comparison: DONUT vs TrOCR + YOLO

## Project Structure
```
receipt_ocr_comparison/
├── 00_project_structure.md       ← This file
├── 01_dataset_preparation.py     ← Dataset loading & preprocessing
├── 02_train_donut.py             ← DONUT fine-tuning
├── 03_train_trocr_yolo.py        ← TrOCR + YOLO fine-tuning
├── 04_evaluate.py                ← Unified evaluation & metrics
├── 05_compare_results.py         ← Visualization & comparison report
└── requirements.txt              ← All dependencies
```

## Quick Start
```bash
pip install -r requirements.txt

# 1. Prepare dataset
python 01_dataset_preparation.py

# 2. Train models
python 02_train_donut.py
python 03_train_trocr_yolo.py

# 3. Evaluate & compare
python 04_evaluate.py
python 05_compare_results.py
```
