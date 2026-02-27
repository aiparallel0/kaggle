# compare_approaches.py — Usage Guide

Compares two receipt KIE approaches head-to-head:

| | Approach A | Approach B |
|---|---|---|
| **Model** | YOLOv8 (`best.pt`) + TrOCR (`trocr-base-printed`) | DONUT (`donut-base-finetuned-cord-v2`) |
| **Method** | Detect regions → OCR crops → Regex extraction | End-to-end vision-language generation |
| **Fields** | company, date, address, total | company, date, address, total |

---

## Install

```bash
pip install -r requirements_compare.txt
```

---

## Label File Format

Each image needs a matching `.txt` file with the **same stem name**:

```
# Tab-separated  (recommended)
company    MYDIN MALL (M) SDN BHD
date       25/12/2023
address    NO 1, JALAN PUCHONG 47100
total      47.80

# Or colon-separated
company: MYDIN MALL (M) SDN BHD
date: 25/12/2023

# Or JSON
{"company": "MYDIN MALL", "date": "25/12/2023", "address": "...", "total": "47.80"}
```

Directory layout:
```
images/
  receipt_001.jpg
  receipt_002.jpg
labels/
  receipt_001.txt
  receipt_002.txt
```

---

## Run

```bash
# Full comparison (both approaches)
python compare_approaches.py \
    --images  ./images \
    --labels  ./labels \
    --yolo    best.pt \
    --donut   naver-clova-ix/donut-base-finetuned-cord-v2

# Your fine-tuned DONUT checkpoint
python compare_approaches.py \
    --images  ./images \
    --labels  ./labels \
    --yolo    best.pt \
    --donut   ./models/donut_sroie_exp1

# Quick test on 10 images
python compare_approaches.py \
    --images ./images --labels ./labels \
    --yolo best.pt --max-images 10

# DONUT only (if best.pt not available)
python compare_approaches.py \
    --images ./images --labels ./labels \
    --skip-yolo

# YOLOv8+TrOCR only
python compare_approaches.py \
    --images ./images --labels ./labels \
    --skip-donut
```

---

## Output

**Console** — printed table per approach + side-by-side comparison:
```
════════════════════════════════════════
  YOLOV8+TROCR+REGEX
════════════════════════════════════════
  Images evaluated : 63  (failed: 0)
  Mean latency     : 312.4 ± 45.1 ms/image
  Throughput       : 3.20 images/sec

  Field        Precision    Recall        F1  Accuracy   Lat(ms)
  ──────────────────────────────────────────────────────────────
  company         0.8500    0.8200    0.8347    0.8200     312.4
  date            0.9200    0.9100    0.9150    0.9100     312.4
  address         0.7100    0.6800    0.6947    0.6800     312.4
  total           0.9400    0.9300    0.9350    0.9300     312.4
  ──────────────────────────────────────────────────────────────
  GLOBAL                              0.8449    0.8350     312.4
```

**Files saved:**
- `comparison_results.json` — full per-image predictions + metrics
- `plots/comparison_f1.png` — per-field F1 grouped bar chart
- `plots/comparison_speed_accuracy.png` — speed vs accuracy scatter
- `plots/comparison_overall.png` — global F1 + accuracy bars

---

## Notes

- **Matching:** Uses exact match after normalisation (lowercase, whitespace collapse). Address field also accepts substring match to handle OCR truncation.
- **YOLO fallback:** If `best.pt` fails to load, falls back to horizontal strip crops so TrOCR+Regex still runs.
- **DONUT prompt:** Auto-detects SROIE (`<s_sroie>`) vs CORD (`<s_cord-v2>`) checkpoint format.
- **CPU vs GPU:** GPU strongly recommended. DONUT on CPU is ~10–20× slower.
