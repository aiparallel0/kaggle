"""Performance monitoring and metrics tracking."""

import time
import threading
from pathlib import Path
from typing import Dict
from collections import defaultdict


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.lock = threading.Lock()
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str):
        """End timing and record duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            with self.lock:
                self.metrics[f"{operation}_duration"].append(duration)
            del self.start_times[operation]
            return duration
        return 0.0
    
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value."""
        with self.lock:
            self.metrics[metric_name].append(value)
    
    def get_stats(self, metric_name: str) -> Dict:
        """Get statistics for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = self.metrics[metric_name]
        try:
            import numpy as np
            return {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'total': np.sum(values)
            }
        except ImportError:
            # Fallback without numpy
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'total': sum(values)
            }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all metrics."""
        return {metric: self.get_stats(metric) for metric in self.metrics.keys()}
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.metrics.clear()
            self.start_times.clear()
    
    def export_to_csv(self, filepath: Path):
        """Export metrics to CSV."""
        try:
            import pandas as pd
            stats = self.get_all_stats()
            df = pd.DataFrame(stats).T
            df.to_csv(filepath)
            print(f"Metrics exported to {filepath}")
        except Exception as e:
            print(f"Failed to export metrics: {e}")
