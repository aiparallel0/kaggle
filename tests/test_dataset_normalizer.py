"""Tests for dataset_normalizer.DatasetNormalizer."""

from pathlib import Path

import pytest

from dataset_normalizer import DatasetNormalizer, normalise_samples


def _make_samples(gts: list[dict]) -> list[tuple[Path, dict]]:
    """Build a list of (Path, dict) samples from a list of raw gt dicts."""
    return [(Path(f"/fake/{i}.jpg"), gt) for i, gt in enumerate(gts)]


class TestAliasResolution:
    def test_store_name_maps_to_company(self):
        samples = _make_samples(
            [
                {
                    "store_name": "ACME Corp",
                    "date": "01/01/2024",
                    "address": "123 St",
                    "total": "10.00",
                }
            ]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == "ACME Corp"

    def test_merchant_maps_to_company(self):
        samples = _make_samples(
            [{"merchant": "My Shop", "date": "01/01/2024", "address": "Addr", "total": "5.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == "My Shop"

    def test_grand_total_maps_to_total(self):
        samples = _make_samples(
            [{"company": "Shop", "date": "01/01/2024", "address": "Addr", "grand_total": "99.99"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["total"] == "99.99"

    def test_receipt_date_maps_to_date(self):
        samples = _make_samples(
            [{"company": "Shop", "receipt_date": "12/12/2023", "address": "Addr", "total": "1.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["date"] == "12/12/2023"


class TestNAHandling:
    def test_na_becomes_empty_string(self):
        samples = _make_samples(
            [{"company": "N/A", "date": "01/01/2024", "address": "Addr", "total": "1.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == ""

    def test_lowercase_na_becomes_empty_string(self):
        samples = _make_samples(
            [{"company": "n/a", "date": "01/01/2024", "address": "Addr", "total": "1.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == ""

    def test_null_becomes_empty_string(self):
        samples = _make_samples(
            [{"company": "null", "date": "01/01/2024", "address": "Addr", "total": "1.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == ""

    def test_none_value_becomes_empty_string(self):
        samples = _make_samples(
            [{"company": None, "date": "01/01/2024", "address": "Addr", "total": "1.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == ""


class TestMissingKeyDefaulting:
    def test_missing_keys_default_to_empty_string(self):
        """A dict with only 2 fields should produce all 4 canonical keys."""
        samples = _make_samples([{"company": "ACME", "date": "01/01/2024"}])
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        gt = result[0][1]
        assert set(gt.keys()) == {"company", "date", "address", "total"}
        assert gt["address"] == ""
        assert gt["total"] == ""

    def test_empty_input_dict_gives_all_four_keys(self):
        samples = _make_samples([{}])
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        gt = result[0][1]
        assert set(gt.keys()) == {"company", "date", "address", "total"}
        for v in gt.values():
            assert v == ""


class TestCoverageCheck:
    def _full_sample(self, company="ACME", date="01/01/2024", total="1.00"):
        return {"company": company, "date": date, "address": "123 St", "total": total}

    def test_raises_when_coverage_below_threshold(self):
        """10 samples where company is empty → coverage=0% < 30% threshold."""
        gts = [{"date": "01/01/2024", "address": "Addr", "total": "1.00"} for _ in range(10)]
        samples = _make_samples(gts)
        with pytest.raises(ValueError, match="company"):
            DatasetNormalizer(coverage_threshold=0.30).normalise(samples)

    def test_passes_when_coverage_above_threshold(self):
        """10 samples all having company → coverage=100% ≥ 30% threshold."""
        gts = [self._full_sample() for _ in range(10)]
        samples = _make_samples(gts)
        result = DatasetNormalizer(coverage_threshold=0.30).normalise(samples)
        assert len(result) == 10

    def test_disabled_coverage_check(self):
        """coverage_threshold=0.0 disables the check entirely."""
        gts = [{"date": "01/01/2024", "address": "Addr", "total": "1.00"} for _ in range(10)]
        samples = _make_samples(gts)
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert len(result) == 10


class TestCurrencyStripping:
    def _sample(self, total: str) -> list[tuple[Path, dict]]:
        return _make_samples(
            [{"company": "Shop", "date": "01/01/2024", "address": "Addr", "total": total}]
        )

    def test_strip_rm_prefix(self):
        result = DatasetNormalizer(coverage_threshold=0.0, strip_currency=True).normalise(
            self._sample("RM 47.80")
        )
        assert result[0][1]["total"] == "47.80"

    def test_strip_dollar_prefix(self):
        result = DatasetNormalizer(coverage_threshold=0.0, strip_currency=True).normalise(
            self._sample("$10.50")
        )
        assert result[0][1]["total"] == "10.50"

    def test_strip_s_dollar_prefix(self):
        result = DatasetNormalizer(coverage_threshold=0.0, strip_currency=True).normalise(
            self._sample("S$25.00")
        )
        assert result[0][1]["total"] == "25.00"

    def test_no_strip_when_disabled(self):
        result = DatasetNormalizer(coverage_threshold=0.0, strip_currency=False).normalise(
            self._sample("RM 47.80")
        )
        assert result[0][1]["total"] == "RM 47.80"

    def test_no_strip_when_disabled_dollar(self):
        result = DatasetNormalizer(coverage_threshold=0.0, strip_currency=False).normalise(
            self._sample("$10.50")
        )
        assert result[0][1]["total"] == "$10.50"


class TestConvenienceWrapper:
    def test_normalise_samples_returns_same_as_class(self):
        samples = _make_samples(
            [{"store_name": "X", "date": "01/01/2024", "address": "A", "total": "1.00"}]
        )
        result = normalise_samples(samples, source_name="test", coverage_threshold=0.0)
        assert result[0][1]["company"] == "X"
