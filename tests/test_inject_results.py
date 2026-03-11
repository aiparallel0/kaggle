"""Tests for inject_results.py — _safe helper, PaperInjector, var map building."""

import json
import tempfile
from pathlib import Path

import pytest

from inject_results import (
    EXP_NAMES,
    LEADERBOARD,
    PaperInjector,
    UnresolvedVarError,
    _safe,
)

# ---------------------------------------------------------------------------
# _safe helper
# ---------------------------------------------------------------------------


class TestSafe:
    def test_present_value(self):
        assert _safe({"f1": 0.9123}, "f1") == "0.9123"

    def test_missing_key(self):
        assert _safe({}, "f1") == "N/A"

    def test_none_value(self):
        assert _safe({"f1": None}, "f1") == "N/A"

    def test_custom_format(self):
        assert _safe({"f1": 0.9123}, "f1", fmt=".2f") == "0.91"

    def test_zero_value(self):
        assert _safe({"f1": 0.0}, "f1") == "0.0000"

    def test_integer_value(self):
        assert _safe({"n": 500}, "n", fmt="d") == "500"


# ---------------------------------------------------------------------------
# EXP_NAMES & LEADERBOARD
# ---------------------------------------------------------------------------


class TestStaticData:
    def test_exp_names_has_at_least_eight_entries(self):
        assert len(EXP_NAMES) >= 8

    def test_exp_names_keys_include_1_to_8(self):
        assert {str(i) for i in range(1, 9)}.issubset(set(EXP_NAMES.keys()))

    def test_exp_names_keys_include_9_to_18(self):
        # Architecture comparison experiments 9-18 should be present
        assert {str(i) for i in range(9, 19)}.issubset(set(EXP_NAMES.keys()))

    def test_leaderboard_has_entries(self):
        assert len(LEADERBOARD) > 0

    def test_leaderboard_scores_are_valid(self):
        for name, score in LEADERBOARD:
            assert isinstance(name, str)
            assert 0 < score <= 1.0, f"Invalid score for {name}: {score}"


# ---------------------------------------------------------------------------
# PaperInjector
# ---------------------------------------------------------------------------


class TestPaperInjector:
    def _make_injector(self, tmp_dir, all_exp=None, eval_res=None, template=""):
        """Create a PaperInjector with mock data."""
        results_dir = Path(tmp_dir) / "results"
        results_dir.mkdir()

        if all_exp is not None:
            (results_dir / "all_experiments.json").write_text(json.dumps(all_exp), encoding="utf-8")
        if eval_res is not None:
            (results_dir / "evaluation_results.json").write_text(
                json.dumps(eval_res), encoding="utf-8"
            )

        template_path = Path(tmp_dir) / "paper.tex"
        template_path.write_text(template, encoding="utf-8")

        return PaperInjector(results_dir, template_path)

    def test_build_var_map_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp)
            var_map = injector.build_var_map()
            assert "best_f1" in var_map
            assert var_map["best_f1"] == "0.0000"

    def test_build_var_map_with_experiments(self):
        all_exp = {
            "1": {
                "metrics": {"global_f1": 0.85, "global_precision": 0.86, "global_recall": 0.84},
                "num_train_samples": 500,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, all_exp=all_exp)
            var_map = injector.build_var_map()
            assert var_map["exp1_f1"] == "0.8500"
            assert var_map["exp1_prec"] == "0.8600"
            assert var_map["exp1_n"] == "500"

    def test_best_experiment_selection(self):
        all_exp = {
            "1": {"metrics": {"global_f1": 0.80}, "num_train_samples": 500},
            "2": {"metrics": {"global_f1": 0.90}, "num_train_samples": 1000},
            "3": {"metrics": {"global_f1": 0.85}, "num_train_samples": 800},
        }
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, all_exp=all_exp)
            var_map = injector.build_var_map()
            assert var_map["best_exp"] == "2"
            assert var_map["best_f1"] == "0.9000"

    def test_fill_substitution(self):
        all_exp = {
            "1": {"metrics": {"global_f1": 0.85}, "num_train_samples": 500},
        }
        template = r"F1 = \VAR{exp1_f1}, Best = \VAR{best_f1}"
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, all_exp=all_exp, template=template)
            filled = injector.fill()
            assert "0.8500" in filled
            assert r"\VAR{" not in filled

    def test_fill_raises_on_unresolved(self):
        template = r"\VAR{nonexistent_var}"
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, template=template)
            try:
                injector.fill(strict=True)
                assert False, "Should have raised UnresolvedVarError"
            except UnresolvedVarError as e:
                assert "nonexistent_var" in str(e)

    def test_pretrained_metrics(self):
        eval_res = {
            "pretrained_metrics": {
                "global_f1": 0.10,
                "global_precision": 0.12,
                "global_recall": 0.09,
                "overall_exact_match": 0.05,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, eval_res=eval_res)
            var_map = injector.build_var_map()
            assert var_map["pre_f1"] == "0.1000"
            assert var_map["pre_f1_pct"] == "10.00"

    def test_gain_calculation(self):
        all_exp = {
            "1": {"metrics": {"global_f1": 0.80}, "num_train_samples": 500},
            "4": {"metrics": {"global_f1": 0.85}, "num_train_samples": 1400},
        }
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, all_exp=all_exp)
            var_map = injector.build_var_map()
            assert var_map["gain_1_4"] == "+0.0500"


# ---------------------------------------------------------------------------
# Partial-results robustness (paper/presentation generation)
# ---------------------------------------------------------------------------


class TestPartialResultsRobustness:
    """Paper/presentation generation must not fail when only some experiments
    have completed.  Unresolved \\VAR{} vars should fall back to "---"."""

    def _make_injector(self, tmp, all_exp=None, template=None, eval_res=None):
        tmp = Path(tmp)
        if all_exp:
            (tmp / "all_experiments.json").write_text(json.dumps(all_exp), encoding="utf-8")
        if eval_res:
            (tmp / "evaluation_results.json").write_text(json.dumps(eval_res), encoding="utf-8")
        tmpl_path = tmp / "template.tex"
        if template:
            tmpl_path.write_text(template, encoding="utf-8")
        else:
            tmpl_path.write_text(r"F1=\VAR{exp1_f1}", encoding="utf-8")
        return PaperInjector(tmp, tmpl_path)

    def test_fill_with_no_results_does_not_raise(self):
        """With zero experiment results, fill() must return a string, not raise."""
        template = r"\VAR{exp1_f1} \VAR{exp2_f1} \VAR{exp3_f1} \VAR{best_f1}"
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, template=template)
            import warnings as _w

            with _w.catch_warnings(record=True):
                _w.simplefilter("always")
                result = injector.fill()
            assert isinstance(result, str), "fill() must return a string"
            assert r"\VAR{" not in result, "No raw \\VAR{} must remain in output"

    def test_fill_with_partial_results_does_not_raise(self):
        """With only Exp 1 complete, fill() for a template that references all
        8 experiments must not raise — missing ones fall back to '---'."""
        all_exp = {
            "1": {
                "metrics": {
                    "global_f1": 0.85,
                    "global_precision": 0.86,
                    "global_recall": 0.84,
                    "overall_exact_match": 0.75,
                    "company_f1": 0.90,
                    "company_ned": 0.05,
                    "date_f1": 0.95,
                    "date_ned": 0.02,
                    "address_f1": 0.80,
                    "address_ned": 0.10,
                    "total_f1": 0.85,
                    "total_ned": 0.08,
                },
                "num_train_samples": 500,
            }
        }
        # Template references exp2_f1 through exp8_f1 (not in results)
        template = " ".join(rf"\VAR{{exp{i}_f1}}" for i in range(1, 9))
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, all_exp=all_exp, template=template)
            import warnings as _w

            with _w.catch_warnings(record=True):
                _w.simplefilter("always")
                result = injector.fill()
            assert r"\VAR{" not in result, "No raw \\VAR{} must remain in output"
            assert "0.8500" in result, "Exp 1 F1 must be filled from results"
            assert "---" in result, "Missing exps must fall back to '---'"

    def test_fill_strict_raises_for_unknown_var(self):
        """fill(strict=True) must still raise UnresolvedVarError for truly
        unknown variables (not in pre-populated fallbacks)."""
        template = r"\VAR{completely_unknown_key_xyz}"
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, template=template)
            with pytest.raises(UnresolvedVarError):
                injector.fill(strict=True)

    def test_all_exp_vars_pre_populated(self):
        """build_var_map() must pre-populate expN_* vars for all N=1..8
        even when no experiments have completed."""
        from constants import FIELDS

        with tempfile.TemporaryDirectory() as tmp:
            template_path = Path(tmp) / "t.tex"
            template_path.write_text("x", encoding="utf-8")
            injector = PaperInjector(Path(tmp), template_path)
            var_map = injector.build_var_map()
            for i in range(1, 9):
                assert f"exp{i}_n" in var_map, f"exp{i}_n missing from var_map"
                assert f"exp{i}_f1" in var_map, f"exp{i}_f1 missing from var_map"
                for fld in FIELDS:
                    assert f"exp{i}_{fld}_f1" in var_map, f"exp{i}_{fld}_f1 missing from var_map"
