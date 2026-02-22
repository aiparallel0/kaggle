"""Before/after comparison visualizations for OCR receipt analysis."""

from pathlib import Path
from typing import Dict


def create_comparison_chart(before_data: Dict, after_data: Dict, output_path: Path):
    """Create before/after comparison chart.
    
    Args:
        before_data: Dictionary with 'accuracy', 'speed', 'completeness' keys
        after_data: Dictionary with 'accuracy', 'speed', 'completeness' keys
        output_path: Path where to save the comparison chart
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Performance Comparison: Before vs After', 
                    fontsize=16, fontweight='bold')
        
        # Metrics to compare
        metrics = ['Accuracy', 'Processing Speed', 'Data Completeness']
        before_values = [
            before_data.get('accuracy', 0.7),
            before_data.get('speed', 0.5),
            before_data.get('completeness', 0.6)
        ]
        after_values = [
            after_data.get('accuracy', 0.9),
            after_data.get('speed', 0.8),
            after_data.get('completeness', 0.85)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Before/After comparison chart
        axes[0].bar(x - width/2, before_values, width, label='Before', color='lightcoral')
        axes[0].bar(x + width/2, after_values, width, label='After', color='lightgreen')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Metric Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].legend()
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)
        
        # Improvement percentages
        improvements = [(after - before) / before * 100 if before > 0 else 0
                       for before, after in zip(before_values, after_values)]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[1].barh(metrics, improvements, color=colors, alpha=0.7)
        axes[1].set_xlabel('Improvement (%)')
        axes[1].set_title('Percentage Improvement')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (metric, imp) in enumerate(zip(metrics, improvements)):
            axes[1].text(imp, i, f'{imp:+.1f}%', va='center', 
                        ha='left' if imp > 0 else 'right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison chart saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating comparison chart: {e}")
        import traceback
        traceback.print_exc()
        return False
