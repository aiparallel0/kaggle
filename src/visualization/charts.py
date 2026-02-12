"""Visualization and chart generation for OCR receipt analysis."""

from pathlib import Path
from typing import Dict, List, Union
from collections import Counter


class Visualizer:
    """Create charts and visualizations."""
    
    @staticmethod
    def plot_processing_stats(receipts: List, output_path: Path):
        """Create comprehensive statistics charts."""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from ..utils.models import ReceiptData
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Receipt Processing Statistics', fontsize=16, fontweight='bold')
            
            # Extract data
            types = [r.receipt_type.value for r in receipts]
            confidences = [r.confidence for r in receipts]
            processing_times = [r.processing_time for r in receipts]
            totals = [r.total for r in receipts if r.total]
            dates = [r.date for r in receipts if r.date]
            stores = [r.store_name for r in receipts if r.store_name]
            
            # 1. Receipt types distribution
            type_counts = Counter(types)
            axes[0, 0].bar(type_counts.keys(), type_counts.values(), color='skyblue')
            axes[0, 0].set_title('Receipt Types Distribution')
            axes[0, 0].set_xlabel('Type')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Confidence distribution
            axes[0, 1].hist(confidences, bins=20, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Classification Confidence')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.mean(confidences), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(confidences):.2f}')
            axes[0, 1].legend()
            
            # 3. Processing time distribution
            axes[0, 2].hist(processing_times, bins=20, color='lightcoral', edgecolor='black')
            axes[0, 2].set_title('Processing Time Distribution')
            axes[0, 2].set_xlabel('Time (seconds)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].axvline(np.mean(processing_times), color='blue', linestyle='--',
                              label=f'Mean: {np.mean(processing_times):.2f}s')
            axes[0, 2].legend()
            
            # 4. Total amounts distribution
            if totals:
                axes[1, 0].hist(totals, bins=20, color='gold', edgecolor='black')
                axes[1, 0].set_title('Receipt Totals Distribution')
                axes[1, 0].set_xlabel('Amount ($)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].axvline(np.mean(totals), color='red', linestyle='--',
                                  label=f'Mean: ${np.mean(totals):.2f}')
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'No total data', ha='center', va='center')
            
            # 5. Top stores
            if stores:
                store_counts = Counter(stores)
                top_stores = dict(store_counts.most_common(10))
                axes[1, 1].barh(list(top_stores.keys()), list(top_stores.values()), 
                               color='plum')
                axes[1, 1].set_title('Top 10 Stores')
                axes[1, 1].set_xlabel('Count')
                axes[1, 1].set_ylabel('Store')
            else:
                axes[1, 1].text(0.5, 0.5, 'No store data', ha='center', va='center')
            
            # 6. Date distribution
            if dates:
                date_counts = Counter(dates)
                sorted_dates = sorted(date_counts.keys())
                counts = [date_counts[d] for d in sorted_dates]
                axes[1, 2].plot(range(len(sorted_dates)), counts, marker='o', color='teal')
                axes[1, 2].set_title('Receipts Over Time')
                axes[1, 2].set_xlabel('Date Index')
                axes[1, 2].set_ylabel('Count')
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'No date data', ha='center', va='center')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Statistics chart saved to {output_path}")
        except Exception as e:
            print(f"Error creating statistics chart: {e}")
            raise
    
    @staticmethod
    def create_annotated_receipt(image_path: Union[str, Path], 
                                 receipt_data,
                                 output_path: Path):
        """Create annotated receipt image with extracted data."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw annotations
            y_offset = 10
            annotations = [
                f"Store: {receipt_data.store_name or 'N/A'}",
                f"Date: {receipt_data.date or 'N/A'}",
                f"Total: ${receipt_data.total or 0:.2f}",
                f"Type: {receipt_data.receipt_type.value}",
                f"Confidence: {receipt_data.confidence:.2%}"
            ]
            
            for annotation in annotations:
                # Draw background rectangle
                bbox = draw.textbbox((10, y_offset), annotation, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0, 180))
                draw.text((10, y_offset), annotation, fill=(255, 255, 0), font=font)
                y_offset += 30
            
            # Save
            image.save(output_path, quality=95)
            print(f"Annotated image saved to {output_path}")
        except Exception as e:
            print(f"Error creating annotated receipt: {e}")
            raise
