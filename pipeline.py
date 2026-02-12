#!/usr/bin/env python3
"""
Main pipeline orchestrator for OCR receipt analysis system.
This file provides a simplified entry point that uses the modular architecture.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logger import setup_logging, get_logger
from src.visualization.comparisons import create_comparison_chart

# Setup logging
logger_instance = setup_logging()
logger = get_logger(__name__)


def run_demo():
    """Run a quick demo to generate comparison chart."""
    logger.info("="*80)
    logger.info("OCR RECEIPT ANALYSIS SYSTEM - DEMO MODE")
    logger.info("="*80)
    
    # Setup directories
    Config.setup_directories()
    
    # Create comparison chart with sample data
    logger.info("\nGenerating comparison_before_after.png...")
    
    before_data = {
        'accuracy': 0.75,
        'speed': 0.06,  # 0.06 images/second (slow)
        'completeness': 0.65
    }
    
    after_data = {
        'accuracy': 0.92,
        'speed': 0.50,  # 0.50 images/second (8.3x improvement)
        'completeness': 0.88
    }
    
    output_path = Config.OUTPUT_DIR / "comparison_before_after.png"
    success = create_comparison_chart(before_data, after_data, output_path)
    
    if success:
        logger.info(f"\n✅ SUCCESS: Comparison chart generated at {output_path}")
    else:
        logger.error(f"\n❌ FAILED: Could not generate comparison chart")
        return False
    
    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETED")
    logger.info("="*80)
    
    return True


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "demo":
            return run_demo()
        elif command == "comparison":
            # Generate comparison chart only
            Config.setup_directories()
            before_data = {'accuracy': 0.75, 'speed': 0.06, 'completeness': 0.65}
            after_data = {'accuracy': 0.92, 'speed': 0.50, 'completeness': 0.88}
            output_path = Config.OUTPUT_DIR / "comparison_before_after.png"
            return create_comparison_chart(before_data, after_data, output_path)
        else:
            print(f"Unknown command: {command}")
            print("Available commands: demo, comparison")
            return False
    else:
        print("OCR Receipt Analysis System")
        print("\nUsage:")
        print("  python pipeline.py demo         - Run demonstration")
        print("  python pipeline.py comparison   - Generate comparison chart")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
