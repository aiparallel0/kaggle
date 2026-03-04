"""Tests for address_extractor.py and multi-line SROIE key file parsing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from address_extractor import extract_address_from_seller

# ---------------------------------------------------------------------------
# extract_address_from_seller
# ---------------------------------------------------------------------------


class TestExtractAddressFromSeller:
    def test_street_number_split(self):
        """Typical Invoices-DONUT seller string with street number."""
        company, address = extract_address_from_seller(
            "Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228"
        )
        assert "Patel" in company
        assert "Montgomery" in company
        assert "356" in address
        assert "Kyle Vista" in address

    def test_po_box(self):
        """P.O. Box pattern detected and split correctly."""
        company, address = extract_address_from_seller("ACME Corp P.O. Box 1234 Springfield")
        assert company == "ACME Corp"
        assert "Box 1234" in address

    def test_street_suffix_only(self):
        """Street suffix like 'Avenue' detected even without a leading number."""
        company, address = extract_address_from_seller("Smith & Sons Oak Avenue Suite 10")
        assert "Smith" in company
        assert "Avenue" in address

    def test_state_zip_only(self):
        """State abbreviation + ZIP at the end still detected."""
        company, address = extract_address_from_seller("Global Inc CA 90210")
        assert "Global" in company
        assert "CA 90210" in address

    def test_no_address_detected(self):
        """Seller string with no address signals → full string as company."""
        company, address = extract_address_from_seller("Acme Corporation")
        assert company == "Acme Corporation"
        assert address == ""

    def test_empty_string(self):
        company, address = extract_address_from_seller("")
        assert company == ""
        assert address == ""

    def test_whitespace_only(self):
        company, address = extract_address_from_seller("   ")
        assert company == ""
        assert address == ""

    def test_company_trailing_comma_stripped(self):
        """Trailing comma from company name is stripped."""
        company, address = extract_address_from_seller("Johnson, Lee, 100 Main St City, TX 75001")
        # trailing comma/space should be stripped from company
        assert not company.endswith(",")
        assert "100 Main" in address

    def test_returns_tuple_of_strings(self):
        result = extract_address_from_seller("Some Co 123 Road St")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# SROIE key file multi-line address parsing
# ---------------------------------------------------------------------------


class TestSROIEKeyFileParsing:
    """Tests for _load_key_file (dataset_loaders) and _parse_txt_key (train)."""

    def _write_key_file(self, tmp_path: Path, lines: list[str]) -> Path:
        key_file = tmp_path / "X00016469671.txt"
        key_file.write_text("\n".join(lines), encoding="utf-8")
        return key_file

    def test_four_line_file(self, tmp_path):
        """Standard 4-line SROIE key file parses correctly."""
        from dataset_loaders import _load_key_file

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        (key_dir / "sample.txt").write_text(
            "ACME STORE\n25/12/2023\n1 ORCHARD ROAD\n12.50", encoding="utf-8"
        )
        gt = _load_key_file(key_dir, "sample")
        assert gt["company"] == "ACME STORE"
        assert gt["date"] == "25/12/2023"
        assert gt["address"] == "1 ORCHARD ROAD"
        assert gt["total"] == "12.50"

    def test_five_line_file_multi_line_address(self, tmp_path):
        """5-line key file: address spans lines 2 and 3, total is line 4."""
        from dataset_loaders import _load_key_file

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        (key_dir / "sample.txt").write_text(
            "MYDIN MALL\n25/12/2023\nNO 1 JALAN PUCHONG\n47100 PUCHONG JAYA\n47.80",
            encoding="utf-8",
        )
        gt = _load_key_file(key_dir, "sample")
        assert gt["company"] == "MYDIN MALL"
        assert gt["date"] == "25/12/2023"
        assert gt["address"] == "NO 1 JALAN PUCHONG 47100 PUCHONG JAYA"
        assert gt["total"] == "47.80"

    def test_six_line_file_multi_line_address(self, tmp_path):
        """6-line key file: address spans lines 2-4, total is line 5."""
        from dataset_loaders import _load_key_file

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        (key_dir / "sample.txt").write_text(
            "WATSON'S SODA\n01/01/2024\nBLOCK A\nJALAN MANIS\n56000 KL\n9.90",
            encoding="utf-8",
        )
        gt = _load_key_file(key_dir, "sample")
        assert gt["address"] == "BLOCK A JALAN MANIS 56000 KL"
        assert gt["total"] == "9.90"

    def test_train_parse_txt_key_four_lines(self, tmp_path):
        """_load_key_file handles standard 4-line file."""
        from dataset_loaders import _load_key_file  # noqa: E402, I001

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        key_file = key_dir / "sample.txt"
        key_file.write_text("STORE\n02/02/2024\n100 MAIN ST\n5.00", encoding="utf-8")
        gt = _load_key_file(key_dir, "sample")
        assert gt is not None
        assert gt["address"] == "100 MAIN ST"
        assert gt["total"] == "5.00"

    def test_train_parse_txt_key_multiline_address(self, tmp_path):
        """_load_key_file handles 5-line file with multi-line address."""
        from dataset_loaders import _load_key_file  # noqa: E402, I001

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        key_file = key_dir / "sample.txt"
        key_file.write_text(
            "STORE\n02/02/2024\n100 MAIN ST\nSUITE 5\n5.00", encoding="utf-8"
        )
        gt = _load_key_file(key_dir, "sample")
        assert gt is not None
        assert gt["address"] == "100 MAIN ST SUITE 5"
        assert gt["total"] == "5.00"

    def test_train_parse_txt_key_too_few_lines(self, tmp_path):
        """_load_key_file returns empty dict for files with fewer than 4 lines."""
        from dataset_loaders import _load_key_file  # noqa: E402, I001

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        key_file = key_dir / "sample.txt"
        key_file.write_text("STORE\n02/02/2024\n100 MAIN ST", encoding="utf-8")
        assert _load_key_file(key_dir, "sample") == {}
