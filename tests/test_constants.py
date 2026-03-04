"""Tests for constants.py — validate schema and invariants."""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from constants import BASE_MODEL, EMPTY_GT, FIELDS, IMAGE_EXTS, MAX_LENGTH, NEW_TOKENS, SEED

torch = pytest.importorskip("torch", reason="torch required for masking tests")

from constants import _mask_empty_field_labels  # noqa: E402, I001


class TestFields:
    def test_fields_is_list(self):
        assert isinstance(FIELDS, list)

    def test_fields_has_four_entries(self):
        assert len(FIELDS) == 4

    def test_fields_contains_expected(self):
        assert FIELDS == ["company", "date", "address", "total"]

    def test_fields_are_strings(self):
        for f in FIELDS:
            assert isinstance(f, str)

    def test_fields_are_lowercase(self):
        for f in FIELDS:
            assert f == f.lower()


class TestImageExts:
    def test_image_exts_is_frozenset(self):
        assert isinstance(IMAGE_EXTS, frozenset)

    def test_common_extensions_present(self):
        for ext in (".jpg", ".jpeg", ".png"):
            assert ext in IMAGE_EXTS

    def test_all_start_with_dot(self):
        for ext in IMAGE_EXTS:
            assert ext.startswith(".")

    def test_all_lowercase(self):
        for ext in IMAGE_EXTS:
            assert ext == ext.lower()


class TestMaxLength:
    def test_max_length_is_int(self):
        assert isinstance(MAX_LENGTH, int)

    def test_max_length_positive(self):
        assert MAX_LENGTH > 0

    def test_max_length_value(self):
        assert MAX_LENGTH == 768


class TestBaseModel:
    def test_base_model_is_string(self):
        assert isinstance(BASE_MODEL, str)

    def test_base_model_value(self):
        assert BASE_MODEL == "naver-clova-ix/donut-base"


class TestSeed:
    def test_seed_is_int(self):
        assert isinstance(SEED, int)

    def test_seed_value(self):
        assert SEED == 42


class TestNewTokens:
    def test_new_tokens_is_list(self):
        assert isinstance(NEW_TOKENS, list)

    def test_new_tokens_has_pairs(self):
        # Every opening token should have a matching closing token
        assert len(NEW_TOKENS) % 2 == 0

    def test_sroie_wrapper_present(self):
        assert "<s_sroie>" in NEW_TOKENS
        assert "</s_sroie>" in NEW_TOKENS

    def test_field_tokens_present(self):
        for field in FIELDS:
            assert f"<s_{field}>" in NEW_TOKENS
            assert f"</s_{field}>" in NEW_TOKENS


class TestEmptyGT:
    def test_empty_gt_is_dict(self):
        assert isinstance(EMPTY_GT, dict)

    def test_empty_gt_keys_match_fields(self):
        assert list(EMPTY_GT.keys()) == FIELDS

    def test_empty_gt_values_are_empty_strings(self):
        for v in EMPTY_GT.values():
            assert v == ""


# ---------------------------------------------------------------------------
# _mask_empty_field_labels
# ---------------------------------------------------------------------------

# Token IDs used in mock tokenizer below.
_OPEN_IDS = {"company": 10, "date": 12, "address": 14, "total": 16}
_CLOSE_IDS = {"company": 11, "date": 13, "address": 15, "total": 17}


class _FakeTokenizer:
    """Minimal tokenizer stub for masking tests."""

    unk_token_id = 0

    def convert_tokens_to_ids(self, tok: str) -> int:
        for f, tid in _OPEN_IDS.items():
            if tok == f"<s_{f}>":
                return tid
        for f, tid in _CLOSE_IDS.items():
            if tok == f"</s_{f}>":
                return tid
        return self.unk_token_id


class TestMaskEmptyFieldLabels:
    """Unit tests for _mask_empty_field_labels in constants.py."""

    def _make_labels(self):
        """Return a label tensor covering all four fields with dummy content."""
        # Layout: <s_co> CO </s_co> <s_da> DA </s_da> <s_ad> AD </s_ad> <s_to> TO </s_to>
        return torch.tensor([10, 100, 11, 12, 200, 13, 14, 300, 15, 16, 400, 17, -100])

    def test_empty_address_masked(self):
        labels = self._make_labels()
        gt = {"company": "ACME", "date": "2024", "address": "", "total": "9.99"}
        result = _mask_empty_field_labels(labels, gt, _FakeTokenizer())
        # <s_address>=14 at idx 6, </s_address>=15 at idx 8 → both -100
        assert result[6].item() == -100
        assert result[7].item() == -100
        assert result[8].item() == -100

    def test_non_empty_fields_unchanged(self):
        labels = self._make_labels()
        gt = {"company": "ACME", "date": "2024", "address": "", "total": "9.99"}
        result = _mask_empty_field_labels(labels, gt, _FakeTokenizer())
        # company, date, total spans should be untouched
        assert result[0].item() == 10  # <s_company>
        assert result[1].item() == 100  # company value
        assert result[2].item() == 11  # </s_company>
        assert result[4].item() == 200  # date value
        assert result[10].item() == 400  # total value

    def test_all_empty_all_masked(self):
        labels = self._make_labels()
        gt = {"company": "", "date": "", "address": "", "total": ""}
        result = _mask_empty_field_labels(labels, gt, _FakeTokenizer())
        # All tag and content positions should be -100 (padding was already -100)
        for i in range(12):
            assert result[i].item() == -100, f"Position {i} should be -100"

    def test_no_empty_fields_unchanged(self):
        labels = self._make_labels()
        gt = {"company": "A", "date": "B", "address": "C", "total": "D"}
        original = labels.clone()
        result = _mask_empty_field_labels(labels, gt, _FakeTokenizer())
        assert torch.equal(result, original)

    def test_whitespace_only_treated_as_empty(self):
        labels = self._make_labels()
        gt = {"company": "ACME", "date": "2024", "address": "   ", "total": "9.99"}
        result = _mask_empty_field_labels(labels, gt, _FakeTokenizer())
        assert result[6].item() == -100
        assert result[8].item() == -100

    def test_unknown_token_skipped(self):
        """When a special token is unknown (maps to unk_id), the field is skipped."""
        labels = self._make_labels()
        gt = {"company": "", "date": "2024", "address": "C", "total": "9.99"}

        class AllUnkTokenizer(_FakeTokenizer):
            def convert_tokens_to_ids(self, tok):
                return self.unk_token_id  # everything maps to unk

        original = labels.clone()
        result = _mask_empty_field_labels(labels, gt, AllUnkTokenizer())
        # No masking should occur because all tokens map to unk
        assert torch.equal(result, original)
