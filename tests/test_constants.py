"""Tests for constants.py — validate schema and invariants."""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import BASE_MODEL, EMPTY_GT, FIELDS, IMAGE_EXTS, MAX_LENGTH, NEW_TOKENS, SEED


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
        assert MAX_LENGTH == 512


class TestBaseModel:
    def test_base_model_is_string(self):
        assert isinstance(BASE_MODEL, str)

    def test_base_model_value(self):
        assert BASE_MODEL == "naver-clova-ix/donut-base-finetuned-cord-v2"


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
