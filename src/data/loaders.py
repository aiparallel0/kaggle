"""Data loading utilities for receipt processing."""

import csv
from pathlib import Path
from typing import List, Dict, Tuple


def load_training_data(csv_path: Path) -> Tuple[List[str], List[str]]:
    """Load training data from CSV file.
    
    Args:
        csv_path: Path to CSV file with 'text' and 'label' columns
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'text' in row and 'label' in row:
                    texts.append(row['text'])
                    labels.append(row['label'])
        
        print(f"Loaded {len(texts)} training samples from {csv_path}")
        return texts, labels
    except Exception as e:
        print(f"Error loading training data: {e}")
        return [], []


def save_to_csv(data: List[Dict], output_path: Path):
    """Save data to CSV file.
    
    Args:
        data: List of dictionaries to save
        output_path: Path where to save CSV
    """
    if not data:
        print("No data to save")
        return
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Saved {len(data)} records to {output_path}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")


def load_from_csv(csv_path: Path) -> List[Dict]:
    """Load data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of dictionaries
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        print(f"Loaded {len(data)} records from {csv_path}")
        return data
    except Exception as e:
        print(f"Error loading from CSV: {e}")
        return []
