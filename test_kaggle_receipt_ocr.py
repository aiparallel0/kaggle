#!/usr/bin/env python3
"""
Test suite for kaggle_receipt_ocr.py single-file version.
Tests the process_image function and receipt processing workflow.
"""

import sys
import os
from pathlib import Path
import csv

# Import from the single-file module
import kaggle_receipt_ocr as kro


def test_demo_function():
    """Test that demo function runs successfully."""
    print("[Test 1/5] Testing demo function...")
    try:
        result = kro.run_demo()
        assert result == True, "Demo should return True on success"
        
        # Verify output file was created
        output_file = Path("output/comparison_before_after.png")
        assert output_file.exists(), "Demo should create comparison chart"
        
        print("  ✅ Demo function works correctly")
        return True
    except Exception as e:
        print(f"  ❌ Demo function failed: {e}")
        return False


def test_receipt_parser():
    """Test receipt parsing functionality."""
    print("\n[Test 2/5] Testing receipt parser...")
    try:
        parser = kro.ReceiptParser()
        
        # Test with sample receipt text
        sample_text = """WALMART SUPERCENTER
123 Main Street
Date: 01/15/2024
Milk        $3.99
Bread       $2.50
SUBTOTAL    $6.49
TAX         $0.52
TOTAL       $7.01"""
        
        receipt = parser.parse_receipt(sample_text)
        
        # Verify extracted data
        assert receipt.store_name == "WALMART", f"Expected 'WALMART', got '{receipt.store_name}'"
        assert receipt.date == "01/15/2024", f"Expected '01/15/2024', got '{receipt.date}'"
        assert receipt.total is not None, "Total should be extracted"
        assert receipt.total > 0, "Total should be positive"
        
        print(f"  ✅ Store: {receipt.store_name}")
        print(f"  ✅ Date: {receipt.date}")
        print(f"  ✅ Total: ${receipt.total:.2f}")
        return True
    except Exception as e:
        print(f"  ❌ Parser test failed: {e}")
        return False


def test_process_image_function():
    """Test process_image function with a test image."""
    print("\n[Test 3/5] Testing process_image function...")
    try:
        # Check if test image exists
        test_image = "sample_receipt_annotated.png"
        if not Path(test_image).exists():
            print("  ⚠️  No test image found, skipping image processing test")
            return True
        
        # Test with default output directory
        output_dir = "test_output"
        receipt = kro.process_image(test_image, output_dir)
        
        # Verify receipt object was returned
        assert isinstance(receipt, kro.ReceiptData), "Should return ReceiptData object"
        assert hasattr(receipt, 'processing_time'), "Should have processing_time attribute"
        assert receipt.processing_time > 0, "Processing time should be positive"
        
        # Verify output file was created
        output_files = list(Path(output_dir).glob("receipt_data_*.csv"))
        assert len(output_files) > 0, "Should create output CSV file"
        
        print(f"  ✅ Processed image in {receipt.processing_time:.2f}s")
        print(f"  ✅ Output saved to {output_dir}/")
        return True
    except Exception as e:
        print(f"  ❌ process_image test failed: {e}")
        return False


def test_processor_class():
    """Test ReceiptProcessor class."""
    print("\n[Test 4/5] Testing ReceiptProcessor class...")
    try:
        processor = kro.ReceiptProcessor()
        
        # Verify components are initialized
        assert hasattr(processor, 'ocr_engine'), "Should have ocr_engine"
        assert hasattr(processor, 'parser'), "Should have parser"
        assert hasattr(processor, 'image_processor'), "Should have image_processor"
        
        # Test save_results method
        sample_receipt = kro.ReceiptData(
            store_name="TEST STORE",
            date="2024-01-15",
            total=10.99,
            tax=0.88
        )
        
        output_path = Path("test_output/test_receipt.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processor.save_results(sample_receipt, output_path)
        
        # Verify file was created and contains data
        assert output_path.exists(), "CSV file should be created"
        
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1, "Should have one data row"
            assert rows[0]['store_name'] == "TEST STORE"
            assert rows[0]['total'] == "10.99"
        
        print("  ✅ ReceiptProcessor initialized correctly")
        print("  ✅ save_results works correctly")
        return True
    except Exception as e:
        print(f"  ❌ ReceiptProcessor test failed: {e}")
        return False


def test_main_command_line():
    """Test main function command line interface."""
    print("\n[Test 5/5] Testing main function CLI...")
    try:
        # Save original sys.argv
        original_argv = sys.argv.copy()
        
        # Test demo command
        sys.argv = ["kaggle_receipt_ocr.py", "demo"]
        result = kro.main()
        assert result == True, "Demo command should return True"
        
        # Test comparison command
        sys.argv = ["kaggle_receipt_ocr.py", "comparison"]
        result = kro.main()
        assert result == True, "Comparison command should return True"
        
        # Test invalid command
        sys.argv = ["kaggle_receipt_ocr.py", "invalid_command"]
        result = kro.main()
        assert result == False, "Invalid command should return False"
        
        # Restore original sys.argv
        sys.argv = original_argv
        
        print("  ✅ Demo command works")
        print("  ✅ Comparison command works")
        print("  ✅ Invalid command handling works")
        return True
    except Exception as e:
        # Restore original sys.argv
        sys.argv = original_argv
        print(f"  ❌ CLI test failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test output files."""
    import shutil
    try:
        if Path("test_output").exists():
            shutil.rmtree("test_output")
    except Exception:
        pass


def main():
    """Run all tests."""
    print("="*80)
    print("TESTING KAGGLE_RECEIPT_OCR.PY - SINGLE FILE VERSION")
    print("="*80)
    print()
    
    results = []
    
    # Run tests
    results.append(("Demo Function", test_demo_function()))
    results.append(("Receipt Parser", test_receipt_parser()))
    results.append(("Process Image Function", test_process_image_function()))
    results.append(("ReceiptProcessor Class", test_processor_class()))
    results.append(("Main CLI", test_main_command_line()))
    
    # Clean up
    cleanup_test_files()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
