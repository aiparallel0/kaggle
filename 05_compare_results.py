"""
05_compare_results.py
=====================
Loads results/metrics.json and produces:
  1. Side-by-side bar chart (CER, WER, Macro F1, Latency)
  2. Per-field F1 grouped bar chart
  3. Training loss curves for both models
  4. Printed summary table
  5. results/comparison_report.html  (self-contained)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

RESULTS_DIR   = Path("results")
DONUT_HIST    = Path("models/donut_finetuned/training_history.json")
TROCR_HIST    = Path("models/trocr_finetuned/training_history.json")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"donut": "#4C72B0", "trocr_yolo": "#DD8452"}
LABELS = {"donut": "DONUT", "trocr_yolo": "TrOCR + YOLO"}


# ── Load data ────────────────────────────────────────────────────────────────
def load_metrics():
    with open(RESULTS_DIR / "metrics.json") as f:
        return json.load(f)

def load_history(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Plot 1: Overall metrics bar chart ────────────────────────────────────────
def plot_overall_metrics(metrics: dict, ax=None):
    keys   = ["cer", "wer", "macro_f1"]
    labels = ["CER ↓", "WER ↓", "Macro F1 ↑"]

    x     = np.arange(len(keys))
    width = 0.35

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for i, (model_key, color) in enumerate(COLORS.items()):
        vals = [metrics[model_key][k] for k in keys]
        bars = ax.bar(x + i * width - width/2, vals, width,
                      label=LABELS[model_key], color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Overall OCR & Extraction Metrics")
    ax.legend()
    return ax


# ── Plot 2: Per-field F1 ──────────────────────────────────────────────────────
def plot_field_f1(metrics: dict, ax=None):
    fields = list(next(iter(metrics.values()))["field_metrics"].keys())
    x      = np.arange(len(fields))
    width  = 0.35

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    for i, (model_key, color) in enumerate(COLORS.items()):
        vals = [metrics[model_key]["field_metrics"][f]["f1"] for f in fields]
        bars = ax.bar(x + i * width - width/2, vals, width,
                      label=LABELS[model_key], color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in fields])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Field-Level F1 Score")
    ax.legend()
    return ax


# ── Plot 3: Latency ───────────────────────────────────────────────────────────
def plot_latency(metrics: dict, ax=None):
    models  = list(COLORS.keys())
    means   = [metrics[m]["latency_mean_ms"] for m in models]
    p95s    = [metrics[m]["latency_p95_ms"]  for m in models]
    x       = np.arange(len(models))
    width   = 0.35

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    bars1 = ax.bar(x - width/2, means, width, label="Mean",  color=[COLORS[m] for m in models], alpha=0.85)
    bars2 = ax.bar(x + width/2, p95s,  width, label="P95",   color=[COLORS[m] for m in models], alpha=0.45, hatch="//")

    for bar, val in zip(list(bars1) + list(bars2), means + p95s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models])
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Inference Latency per Image")

    solid  = mpatches.Patch(facecolor="grey", alpha=0.85, label="Mean")
    hatch  = mpatches.Patch(facecolor="grey", alpha=0.45, hatch="//", label="P95")
    ax.legend(handles=[solid, hatch])
    return ax


# ── Plot 4: Training loss curves ─────────────────────────────────────────────
def plot_training_curves(donut_hist, trocr_hist, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    if donut_hist:
        epochs = range(1, len(donut_hist["train_loss"]) + 1)
        ax.plot(epochs, donut_hist["train_loss"], color=COLORS["donut"],     linestyle="-",  label="DONUT train")
        ax.plot(epochs, donut_hist["val_loss"],   color=COLORS["donut"],     linestyle="--", label="DONUT val")

    if trocr_hist:
        epochs = range(1, len(trocr_hist["train_loss"]) + 1)
        ax.plot(epochs, trocr_hist["train_loss"], color=COLORS["trocr_yolo"], linestyle="-",  label="TrOCR train")
        ax.plot(epochs, trocr_hist["val_loss"],   color=COLORS["trocr_yolo"], linestyle="--", label="TrOCR val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    return ax


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary_table(metrics):
    rows = []
    for model_key in COLORS:
        m   = metrics[model_key]
        row = {
            "Model":         LABELS[model_key],
            "CER ↓":         f"{m['cer']:.4f}",
            "WER ↓":         f"{m['wer']:.4f}",
            "Macro F1 ↑":    f"{m['macro_f1']:.4f}",
            "Lat Mean (ms)": f"{m['latency_mean_ms']:.1f}",
            "Lat P95 (ms)":  f"{m['latency_p95_ms']:.1f}",
            "N samples":     m["n_samples"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print(" COMPARISON SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    # Per-field
    for key in next(iter(metrics.values()))["field_metrics"].keys():
        print(f"\n  Field: {key.upper()}")
        for model_key in COLORS:
            fm = metrics[model_key]["field_metrics"][key]
            print(f"    {LABELS[model_key]:15s}  P={fm['precision']:.3f}  R={fm['recall']:.3f}  F1={fm['f1']:.3f}")


# ── HTML Report ───────────────────────────────────────────────────────────────
def generate_html_report(metrics, fig_paths: dict):
    import base64

    def img_tag(path):
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px #0002">'

    rows = []
    for model_key in COLORS:
        m   = metrics[model_key]
        rows.append(f"""
        <tr>
            <td><b>{LABELS[model_key]}</b></td>
            <td>{m['cer']:.4f}</td>
            <td>{m['wer']:.4f}</td>
            <td>{m['macro_f1']:.4f}</td>
            <td>{m['latency_mean_ms']:.1f}</td>
            <td>{m['latency_p95_ms']:.1f}</td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Receipt OCR Comparison Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 1100px; margin: 40px auto; padding: 0 20px; color:#333; }}
  h1   {{ color: #2c3e50; border-bottom: 3px solid #4C72B0; padding-bottom:10px; }}
  h2   {{ color: #34495e; margin-top:40px; }}
  table{{ border-collapse:collapse; width:100%; margin:20px 0; }}
  th   {{ background:#4C72B0; color:#fff; padding:10px 16px; text-align:left; }}
  td   {{ padding:9px 16px; border-bottom:1px solid #eee; }}
  tr:hover {{ background:#f5f8ff; }}
  .grid{{ display:grid; grid-template-columns:1fr 1fr; gap:24px; margin:24px 0; }}
  .card{{ background:#fafafa; border-radius:12px; padding:16px;
          box-shadow:0 2px 8px #0001; }}
</style></head><body>
<h1>Receipt OCR: DONUT vs TrOCR + YOLO — Comparison Report</h1>

<h2>Overall Metrics</h2>
<table>
  <tr><th>Model</th><th>CER ↓</th><th>WER ↓</th><th>Macro F1 ↑</th>
      <th>Latency Mean (ms)</th><th>Latency P95 (ms)</th></tr>
  {''.join(rows)}
</table>

<div class="grid">
  <div class="card"><h3>OCR & Extraction Metrics</h3>{img_tag(fig_paths['overall'])}</div>
  <div class="card"><h3>Inference Latency</h3>{img_tag(fig_paths['latency'])}</div>
  <div class="card"><h3>Per-Field F1</h3>{img_tag(fig_paths['fields'])}</div>
  <div class="card"><h3>Training Loss Curves</h3>{img_tag(fig_paths['curves'])}</div>
</div>

<h2>Methodology</h2>
<p>Dataset: SROIE (626 receipts). Train/val/test split: 70/15/15.
DONUT is evaluated end-to-end (image → JSON). TrOCR+YOLO pipeline:
YOLOv8-medium detects text lines, TrOCR-base-printed transcribes each crop.
Field-level F1 uses exact-match after normalization (lower-case, strip whitespace).
Latency measured on single-image inference (no batching).</p>
</body></html>"""

    out = RESULTS_DIR / "comparison_report.html"
    out.write_text(html)
    print(f"✓ HTML report → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics     = load_metrics()
    donut_hist  = load_history(DONUT_HIST)
    trocr_hist  = load_history(TROCR_HIST)

    print_summary_table(metrics)

    # Create all plots
    fig_paths = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_overall_metrics(metrics, ax)
    path = RESULTS_DIR / "plot_overall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig_paths["overall"] = path

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_field_f1(metrics, ax)
    path = RESULTS_DIR / "plot_fields.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig_paths["fields"] = path

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_latency(metrics, ax)
    path = RESULTS_DIR / "plot_latency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig_paths["latency"] = path

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_training_curves(donut_hist, trocr_hist, ax)
    path = RESULTS_DIR / "plot_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig_paths["curves"] = path

    generate_html_report(metrics, fig_paths)
    print("\nAll done! Check the results/ folder.")
