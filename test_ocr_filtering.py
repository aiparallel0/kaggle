#!/usr/bin/env python3
"""
Focused tests for OCR filtering improvements.
Tests store name filtering, date extraction improvements, and total validation.
"""

import sys
from pathlib import Path

# Import from the single-file module
import kaggle_receipt_ocr as kro


def test_store_name_exclude_words():
    """Test that noise words are excluded from store names."""
    print("[Test 1/6] Testing store name exclude words...")
    try:
        parser = kro.ReceiptParser()
        
        # Test case 1: GST should be excluded
        text_gst = """GST
Registration No: 123456
Some Store Name
Total: $10.00"""
        result = parser.extract_store_name(text_gst)
        # Should skip GST and find "Some Store Name"
        assert result != "GST", f"Should not return 'GST', got '{result}'"
        print(f"  ✅ GST correctly filtered out, got: {result}")
        
        # Test case 2: SECURITY should be excluded
        text_security = """SECURITY
TOTAL AMOUNT
HENG Store
Date: 01/01/2024"""
        result = parser.extract_store_name(text_security)
        assert result != "SECURITY", f"Should not return 'SECURITY', got '{result}'"
        assert result != "TOTAL AMOUNT", f"Should not return 'TOTAL AMOUNT', got '{result}'"
        print(f"  ✅ SECURITY and TOTAL correctly filtered out, got: {result}")
        
        # Test case 3: TAX should be excluded
        text_tax = """TAX INVOICE
RECEIPT NO 123
Store Name
Total: $5.00"""
        result = parser.extract_store_name(text_tax)
        assert result != "TAX INVOICE", f"Should not return 'TAX INVOICE', got '{result}'"
        assert result != "TAX", f"Should not return 'TAX', got '{result}'"
        print(f"  ✅ TAX correctly filtered out, got: {result}")
        
        return True
    except Exception as e:
        print(f"  ❌ Store name exclude words test failed: {e}")
        return False


def test_store_name_trailing_punctuation():
    """Test that trailing punctuation is removed from store names."""
    print("\n[Test 2/6] Testing trailing punctuation removal...")
    try:
        parser = kro.ReceiptParser()
        
        # Test case 1: Trailing comma
        text_comma = """MyTOWN Shopping Centre,
123 Main Street
Date: 01/01/2024
Total: $10.00"""
        result = parser.extract_store_name(text_comma)
        assert result is not None, "Should extract store name"
        assert not result.endswith(','), f"Should remove trailing comma from '{result}'"
        assert result == "MyTOWN Shopping Centre", f"Expected 'MyTOWN Shopping Centre', got '{result}'"
        print(f"  ✅ Trailing comma removed: '{result}'")
        
        # Test case 2: Trailing semicolon
        text_semicolon = """Store Name;
123 Main Street"""
        result = parser.extract_store_name(text_semicolon)
        if result:
            assert not result.endswith(';'), f"Should remove trailing semicolon from '{result}'"
            print(f"  ✅ Trailing semicolon removed: '{result}'")
        
        # Test case 3: Trailing period
        text_period = """Store Name.
123 Main Street"""
        result = parser.extract_store_name(text_period)
        if result:
            assert not result.endswith('.'), f"Should remove trailing period from '{result}'"
            print(f"  ✅ Trailing period removed: '{result}'")
        
        return True
    except Exception as e:
        print(f"  ❌ Trailing punctuation test failed: {e}")
        return False


def test_store_name_length_validation():
    """Test that store names are validated by length (3-40 characters)."""
    print("\n[Test 3/6] Testing store name length validation...")
    try:
        parser = kro.ReceiptParser()
        
        # Test case 1: Too short (< 3 characters)
        text_short = """AB
Long Store Name Here
Date: 01/01/2024"""
        result = parser.extract_store_name(text_short)
        if result:
            assert len(result) >= 3, f"Should reject names < 3 chars, got '{result}'"
        print(f"  ✅ Short names rejected, got: {result}")
        
        # Test case 2: Too long (> 40 characters)
        text_long = """This Is A Very Long Store Name That Exceeds Forty Characters
Date: 01/01/2024"""
        result = parser.extract_store_name(text_long)
        if result:
            assert len(result) <= 40, f"Should reject names > 40 chars, got '{result}' ({len(result)} chars)"
        print(f"  ✅ Long names rejected, got: {result}")
        
        # Test case 3: Valid length
        text_valid = """HENG
123 Main Street
Date: 01/01/2024"""
        result = parser.extract_store_name(text_valid)
        assert result is not None, "Should extract valid store name"
        assert 3 <= len(result) <= 40, f"Store name length should be 3-40, got {len(result)}"
        print(f"  ✅ Valid length accepted: '{result}' ({len(result)} chars)")
        
        return True
    except Exception as e:
        print(f"  ❌ Store name length validation test failed: {e}")
        return False


def test_malaysian_store_recognition():
    """Test that Malaysian stores are recognized."""
    print("\n[Test 4/6] Testing Malaysian store recognition...")
    try:
        parser = kro.ReceiptParser()
        
        # Test known Malaysian stores
        malaysian_stores = [
            ('HENG', 'HENG'),
            ('PASARAYA BORONG PINTAR', 'PASARAYA BORONG PINTAR'),
            ('TAMAN SRI SEGAMBUT', 'TAMAN SRI SEGAMBUT'),
            ('SALON DU CHOCOLAT', 'SALON DU CHOCOLAT'),
        ]
        
        for store_text, expected_store in malaysian_stores:
            text = f"""{store_text}
123 Main Street
Date: 01/01/2024
Total: $10.00"""
            result = parser.extract_store_name(text)
            assert result == expected_store, f"Expected '{expected_store}', got '{result}'"
            print(f"  ✅ Recognized: {expected_store}")
        
        return True
    except Exception as e:
        print(f"  ❌ Malaysian store recognition test failed: {e}")
        return False


def test_total_amount_validation():
    """Test that unreasonable total amounts are rejected."""
    print("\n[Test 5/6] Testing total amount validation...")
    try:
        parser = kro.ReceiptParser()
        
        # Test case 1: Unreasonably high amount (> $10,000)
        text_high = """Store Name
Date: 01/01/2024
TOTAL: $37642.00"""
        result = parser.extract_total(text_high)
        assert result is None, f"Should reject amounts > $10,000, got ${result}"
        print(f"  ✅ High amount rejected: ${37642.00} -> None")
        
        # Test case 2: Valid amount
        text_valid = """Store Name
Date: 01/01/2024
TOTAL: $77.40"""
        result = parser.extract_total(text_valid)
        assert result is not None, "Should accept valid amounts"
        assert 0.01 <= result <= 10000, f"Amount should be in valid range, got ${result}"
        print(f"  ✅ Valid amount accepted: ${result}")
        
        # Test case 3: Amount too low (< $0.01)
        text_low = """Store Name
Date: 01/01/2024
TOTAL: $0.001"""
        result = parser.extract_total(text_low)
        # This might not match the pattern, but if it does, it should be rejected
        if result is not None:
            assert result >= 0.01, f"Should reject amounts < $0.01, got ${result}"
        print(f"  ✅ Low amount handled: ${0.001} -> {result}")
        
        # Test case 4: Multiple amounts, highest is over limit
        text_multiple = """Store Name
Item 1: $10.00
Item 2: $20.00
Item 3: $37642.00
Total: $50.00"""
        result = parser.extract_total(text_multiple)
        # Should find the total pattern first, or reject the high fallback amount
        if result is not None:
            assert result <= 10000, f"Should reject or avoid amounts > $10,000, got ${result}"
        print(f"  ✅ Multiple amounts handled correctly: {result}")
        
        return True
    except Exception as e:
        print(f"  ❌ Total amount validation test failed: {e}")
        return False


def test_improved_date_extraction():
    """Test that additional date patterns are recognized."""
    print("\n[Test 6/6] Testing improved date extraction...")
    try:
        parser = kro.ReceiptParser()
        
        # Test various date formats
        date_tests = [
            ("Date: 01/15/2024", "01/15/2024"),
            ("Date: 01-15-2024", "01-15-2024"),
            ("Date: 01.15.2024", "01.15.2024"),
            ("Date: 2024-01-15", "2024-01-15"),
            ("Date: 2024/01/15", "2024/01/15"),
            ("Date: Jan 15, 2024", "Jan 15, 2024"),
            ("Date: February 28, 2024", "February 28, 2024"),
        ]
        
        for text_snippet, expected_pattern in date_tests:
            text = f"""Store Name
{text_snippet}
Total: $10.00"""
            result = parser.extract_date(text)
            assert result is not None, f"Should extract date from '{text_snippet}', got None"
            print(f"  ✅ Extracted date from '{text_snippet}': {result}")
        
        return True
    except Exception as e:
        print(f"  ❌ Date extraction test failed: {e}")
        return False


def main():
    """Run all focused tests."""
    print("="*80)
    print("TESTING OCR FILTERING IMPROVEMENTS")
    print("="*80)
    print()
    
    results = []
    
    # Run tests
    results.append(("Store Name Exclude Words", test_store_name_exclude_words()))
    results.append(("Store Name Trailing Punctuation", test_store_name_trailing_punctuation()))
    results.append(("Store Name Length Validation", test_store_name_length_validation()))
    results.append(("Malaysian Store Recognition", test_malaysian_store_recognition()))
    results.append(("Total Amount Validation", test_total_amount_validation()))
    results.append(("Improved Date Extraction", test_improved_date_extraction()))
    
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
