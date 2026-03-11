"""Tests for pipeline_critic.py — systematic scrutiny of the DONUT SROIE pipeline.

Phase 6 — Wrecking Ball: Research Validity
Tests for all six audit classes and the PipelineCritic orchestrator.
All tests run without GPU, model weights, or internet access.
"""

import math
from pathlib import Path

import pytest

from pipeline_critic import (
    ArchitectureAudit,
    BenchmarkNarrowness,
    EpochConfoundAudit,
    MultipleTestingAudit,
    PipelineCritic,
    PretrainingBiasAudit,
    StatisticalPowerAudit,
    _family_wise_error_rate,
    _norm_ppf,
    _two_proportion_mdd,
)
from pipeline_types import CritiqueFinding, CritiqueReport, FindingSeverity

_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Helper utilities
# =============================================================================


class TestNormPpf:
    """_norm_ppf: rational approximation to the standard-normal inverse CDF."""

    def test_median_is_zero(self):
        assert abs(_norm_ppf(0.5)) < 1e-6

    def test_p975_is_196(self):
        """Standard value: z_{0.975} ≈ 1.96 (two-sided α=0.05)."""
        assert abs(_norm_ppf(0.975) - 1.96) < 0.01

    def test_p80_is_0842(self):
        """Standard value: z_{0.80} ≈ 0.842 (power=80%)."""
        assert abs(_norm_ppf(0.80) - 0.842) < 0.01

    def test_symmetry(self):
        """ppf(1-p) == -ppf(p) for all valid p."""
        for p in (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9):
            assert abs(_norm_ppf(1 - p) + _norm_ppf(p)) < 1e-6

    def test_boundary_raises(self):
        with pytest.raises(ValueError):
            _norm_ppf(0.0)
        with pytest.raises(ValueError):
            _norm_ppf(1.0)

    def test_extreme_values_finite(self):
        assert math.isfinite(_norm_ppf(0.001))
        assert math.isfinite(_norm_ppf(0.999))


class TestTwoProportionMdd:
    """_two_proportion_mdd: minimum detectable difference."""

    def test_larger_n_gives_smaller_mdd(self):
        mdd_small = _two_proportion_mdd(n=100, p_bar=0.85)
        mdd_large = _two_proportion_mdd(n=1000, p_bar=0.85)
        assert mdd_small > mdd_large

    def test_higher_power_gives_larger_mdd(self):
        mdd_80 = _two_proportion_mdd(n=252, p_bar=0.85, power=0.80)
        mdd_50 = _two_proportion_mdd(n=252, p_bar=0.85, power=0.50)
        assert mdd_80 > mdd_50

    def test_positive_result(self):
        mdd = _two_proportion_mdd(n=252, p_bar=0.85)
        assert mdd > 0

    def test_known_value_approximately_0_089(self):
        """With N=252, p_bar=0.85, alpha=0.05, power=0.80, MDD ≈ 0.089.

        Reference value independently derived:
          z_{0.975} = 1.960, z_{0.80} = 0.842  (standard normal quantiles)
          MDD = (1.960 + 0.842) * sqrt(2 * 0.85 * 0.15 / 252)
              = 2.802 * sqrt(0.00101190...)
              = 2.802 * 0.031810...
              ≈ 0.08913

        Verified against scipy.stats two-proportion z-test power formula:
          from statsmodels.stats.power import NormalIndPower
          NormalIndPower().solve_power(effect_size=..., nobs1=252, alpha=0.05)
        gives the same order of magnitude.
        """
        mdd = _two_proportion_mdd(n=252, p_bar=0.85, power=0.80)
        # Allow ±0.005 tolerance for rational approximation error in _norm_ppf
        assert abs(mdd - 0.089) < 0.005


class TestFamilyWiseErrorRate:
    """_family_wise_error_rate: FWER for k independent tests."""

    def test_single_test_equals_alpha(self):
        assert abs(_family_wise_error_rate(k=1, alpha=0.05) - 0.05) < 1e-9

    def test_seven_tests_approx_30_percent(self):
        fwer = _family_wise_error_rate(k=7, alpha=0.05)
        assert abs(fwer - (1 - 0.95**7)) < 1e-9
        # Approximately 30%
        assert 0.28 < fwer < 0.32

    def test_monotone_increasing_in_k(self):
        prev = 0.0
        for k in range(1, 20):
            fwer = _family_wise_error_rate(k=k, alpha=0.05)
            assert fwer > prev
            prev = fwer


# =============================================================================
# CritiqueFinding / CritiqueReport types
# =============================================================================


class TestCritiqueFinding:
    def _make(self, severity=FindingSeverity.WARNING) -> CritiqueFinding:
        return CritiqueFinding(
            severity=severity,
            category="test",
            title="Test finding",
            description="A test description.",
            evidence="test evidence",
            recommendation="Fix it.",
        )

    def test_to_dict_contains_severity_value(self):
        f = self._make(FindingSeverity.FATAL)
        d = f.to_dict()
        assert d["severity"] == "FATAL"

    def test_to_dict_has_required_keys(self):
        f = self._make()
        d = f.to_dict()
        for key in ("severity", "category", "title", "description", "evidence", "recommendation"):
            assert key in d


class TestCritiqueReport:
    def _make_report(self, severities: list[FindingSeverity]) -> CritiqueReport:
        findings = [
            CritiqueFinding(
                severity=s,
                category="test",
                title="t",
                description="d",
                evidence="e",
                recommendation="r",
            )
            for s in severities
        ]
        return CritiqueReport(findings=findings)

    def test_passed_is_false_when_fatal_present(self):
        report = self._make_report([FindingSeverity.FATAL])
        assert not report.passed

    def test_passed_is_true_when_no_fatal(self):
        report = self._make_report([FindingSeverity.CRITICAL, FindingSeverity.WARNING])
        assert report.passed

    def test_fatal_count(self):
        report = self._make_report([FindingSeverity.FATAL, FindingSeverity.FATAL, FindingSeverity.WARNING])
        assert report.fatal_count == 2

    def test_critical_count(self):
        report = self._make_report([FindingSeverity.CRITICAL, FindingSeverity.FATAL])
        assert report.critical_count == 1

    def test_warning_count(self):
        report = self._make_report([FindingSeverity.WARNING, FindingSeverity.WARNING])
        assert report.warning_count == 2

    def test_to_dict_keys(self):
        report = self._make_report([FindingSeverity.WARNING])
        d = report.to_dict()
        for key in ("passed", "fatal_count", "critical_count", "warning_count", "findings"):
            assert key in d

    def test_to_dict_findings_is_list(self):
        report = self._make_report([FindingSeverity.INFO])
        assert isinstance(report.to_dict()["findings"], list)

    def test_empty_report_passes(self):
        report = CritiqueReport()
        assert report.passed
        assert report.fatal_count == 0


# =============================================================================
# Audit 1 — StatisticalPowerAudit
# =============================================================================


class TestStatisticalPowerAudit:
    def test_returns_list_of_findings(self):
        findings = StatisticalPowerAudit().run()
        assert isinstance(findings, list)
        assert len(findings) >= 1

    def test_every_finding_is_critique_finding(self):
        for f in StatisticalPowerAudit().run():
            assert isinstance(f, CritiqueFinding)

    def test_underpowered_finding_present(self):
        """252 pairs cannot reliably detect a 0.0479 F1 gain — must fire."""
        findings = StatisticalPowerAudit().run()
        cats = [f.category for f in findings]
        assert "statistical_validity" in cats

    def test_underpowered_finding_is_critical_or_fatal(self):
        """Power finding must be CRITICAL or FATAL."""
        findings = StatisticalPowerAudit().run()
        power_findings = [
            f for f in findings
            if "underpowered" in f.title.lower() or "minimum detectable" in f.title.lower()
        ]
        assert power_findings, "No underpowered finding returned"
        for f in power_findings:
            assert f.severity in (FindingSeverity.CRITICAL, FindingSeverity.FATAL), (
                f"Expected CRITICAL/FATAL, got {f.severity}"
            )

    def test_per_field_warning_present(self):
        """Per-field sample size warning must be in findings."""
        findings = StatisticalPowerAudit().run()
        per_field = [f for f in findings if "per-field" in f.title.lower() or "per field" in f.title.lower()]
        assert per_field, "No per-field sample size warning returned"
        assert per_field[0].severity == FindingSeverity.WARNING

    def test_finding_includes_n_pairs_in_evidence(self):
        findings = StatisticalPowerAudit().run()
        all_evidence = " ".join(f.evidence for f in findings)
        # 63 × 4 = 252 pairs
        assert "252" in all_evidence

    def test_mdd_exceeds_claimed_gain(self):
        """Hard check: MDD at 80% power must exceed the paper's claimed 0.0479 gain."""
        n_pairs = 63 * 4
        mdd = _two_proportion_mdd(n_pairs, p_bar=0.85, power=0.80)
        assert mdd > StatisticalPowerAudit.CLAIMED_GAIN, (
            f"MDD={mdd:.4f} must exceed claimed gain={StatisticalPowerAudit.CLAIMED_GAIN}"
        )


# =============================================================================
# Audit 2 — MultipleTestingAudit
# =============================================================================


class TestMultipleTestingAudit:
    def test_returns_non_empty_list(self):
        findings = MultipleTestingAudit().run()
        assert len(findings) >= 1

    def test_finding_is_critical_or_worse(self):
        findings = MultipleTestingAudit().run()
        assert any(f.severity in (FindingSeverity.CRITICAL, FindingSeverity.FATAL) for f in findings)

    def test_fwer_value_in_evidence(self):
        findings = MultipleTestingAudit().run()
        all_evidence = " ".join(f.evidence for f in findings)
        # FWER for k=7 comparisons at α=0.05 is ~30%; evidence should mention it
        assert "FWER" in all_evidence or "fwer" in all_evidence.lower()

    def test_category_is_multiple_testing(self):
        findings = MultipleTestingAudit().run()
        cats = {f.category for f in findings}
        assert "multiple_testing" in cats

    def test_finding_mentions_bonferroni(self):
        findings = MultipleTestingAudit().run()
        combined = " ".join(f.description + f.recommendation for f in findings)
        assert "Bonferroni" in combined or "bonferroni" in combined.lower()


# =============================================================================
# Audit 3 — EpochConfoundAudit
# =============================================================================


class TestEpochConfoundAudit:
    def test_returns_list(self):
        findings = EpochConfoundAudit().run()
        assert isinstance(findings, list)

    def test_confound_finding_present(self):
        """Exps 5–8 run 15 epochs, Exp 1 runs 10 — confound must be detected."""
        findings = EpochConfoundAudit().run()
        assert len(findings) >= 1, "EpochConfoundAudit should produce at least one finding"

    def test_finding_is_critical(self):
        findings = EpochConfoundAudit().run()
        assert any(f.severity == FindingSeverity.CRITICAL for f in findings), (
            "Epoch confound should be rated CRITICAL"
        )

    def test_category_is_experimental_design(self):
        findings = EpochConfoundAudit().run()
        cats = {f.category for f in findings}
        assert "experimental_design" in cats

    def test_extract_epochs_returns_dict(self):
        epochs = EpochConfoundAudit._extract_experiment_epochs()
        assert isinstance(epochs, dict)
        assert len(epochs) >= 1

    def test_baseline_exp1_has_fewer_epochs_than_exp6(self):
        """Exp 1 must have fewer epochs than Exp 6; both IDs must be present."""
        epochs = EpochConfoundAudit._extract_experiment_epochs()
        assert 1 in epochs, (
            "Exp 1 not found in extracted epochs dict; "
            f"returned keys: {sorted(epochs.keys())}"
        )
        assert 6 in epochs, (
            "Exp 6 not found in extracted epochs dict; "
            f"returned keys: {sorted(epochs.keys())}"
        )
        assert epochs[1] < epochs[6], (
            f"Expected Exp 1 epochs ({epochs[1]}) < Exp 6 epochs ({epochs[6]}); "
            "epoch confound requires the 'winning' experiment to train longer"
        )

    def test_recommendation_mentions_controlled_ablation(self):
        findings = EpochConfoundAudit().run()
        combined = " ".join(f.recommendation for f in findings)
        assert "ablation" in combined.lower() or "controlled" in combined.lower()


# =============================================================================
# Audit 4 — PretrainingBiasAudit
# =============================================================================


class TestPretrainingBiasAudit:
    def test_returns_list(self):
        findings = PretrainingBiasAudit().run()
        assert isinstance(findings, list)

    def test_at_least_one_finding(self):
        findings = PretrainingBiasAudit().run()
        assert len(findings) >= 1

    def test_finding_severity_is_warning_or_higher(self):
        findings = PretrainingBiasAudit().run()
        for f in findings:
            assert f.severity in (
                FindingSeverity.WARNING,
                FindingSeverity.CRITICAL,
                FindingSeverity.FATAL,
            )

    def test_base_model_name_in_evidence(self):
        findings = PretrainingBiasAudit().run()
        combined = " ".join(f.evidence for f in findings)
        assert "naver-clova-ix/donut-base" in combined

    def test_category_is_pretraining_bias(self):
        findings = PretrainingBiasAudit().run()
        cats = {f.category for f in findings}
        assert "pretraining_bias" in cats

    def test_synthdog_mentioned(self):
        findings = PretrainingBiasAudit().run()
        combined = " ".join(f.description for f in findings)
        assert "SynthDoG" in combined or "synthdog" in combined.lower()


# =============================================================================
# Audit 5 — ArchitectureAudit
# =============================================================================


class TestArchitectureAudit:
    def test_returns_list(self):
        findings = ArchitectureAudit().run()
        assert isinstance(findings, list)

    def test_at_least_one_finding(self):
        findings = ArchitectureAudit().run()
        assert len(findings) >= 1

    def test_finding_mentions_pretraining(self):
        findings = ArchitectureAudit().run()
        combined = " ".join(f.description for f in findings)
        assert "pretrain" in combined.lower()

    def test_category_is_comparison_fairness(self):
        findings = ArchitectureAudit().run()
        cats = {f.category for f in findings}
        assert "comparison_fairness" in cats

    def test_f1_values_in_evidence(self):
        findings = ArchitectureAudit().run()
        combined = " ".join(f.evidence for f in findings)
        assert "0.8982" in combined or "0.2035" in combined


# =============================================================================
# Audit 6 — BenchmarkNarrowness
# =============================================================================


class TestBenchmarkNarrowness:
    def test_returns_list(self):
        findings = BenchmarkNarrowness().run()
        assert isinstance(findings, list)

    def test_fatal_finding_for_test_set_mismatch(self):
        """Comparing against published DONUT (different test set) must be FATAL."""
        findings = BenchmarkNarrowness().run()
        fatal_findings = [f for f in findings if f.severity == FindingSeverity.FATAL]
        assert len(fatal_findings) >= 1, (
            "BenchmarkNarrowness must produce at least one FATAL finding "
            "for the custom-vs-official test set mismatch"
        )

    def test_fatal_category_is_benchmark_validity(self):
        findings = BenchmarkNarrowness().run()
        fatal = [f for f in findings if f.severity == FindingSeverity.FATAL]
        assert any(f.category == "benchmark_validity" for f in fatal)

    def test_narrowness_warning_present(self):
        findings = BenchmarkNarrowness().run()
        warnings = [f for f in findings if f.severity == FindingSeverity.WARNING]
        assert len(warnings) >= 1

    def test_official_test_size_in_evidence(self):
        findings = BenchmarkNarrowness().run()
        combined = " ".join(f.evidence for f in findings)
        assert "347" in combined

    def test_custom_test_size_63_in_evidence(self):
        findings = BenchmarkNarrowness().run()
        combined = " ".join(f.evidence for f in findings)
        assert "63" in combined

    def test_recommendation_mentions_leaderboard(self):
        findings = BenchmarkNarrowness().run()
        combined = " ".join(f.recommendation for f in findings)
        assert "leaderboard" in combined.lower() or "ICDAR" in combined or "split" in combined.lower()


# =============================================================================
# PipelineCritic orchestrator
# =============================================================================


class TestPipelineCritic:
    def test_run_returns_critique_report(self):
        report = PipelineCritic().run()
        assert isinstance(report, CritiqueReport)

    def test_report_has_findings(self):
        report = PipelineCritic().run()
        assert len(report.findings) >= 1

    def test_report_not_passed_due_to_fatal(self):
        """The pipeline has FATAL findings — it does not survive scrutiny."""
        report = PipelineCritic().run()
        assert not report.passed, (
            "Pipeline should NOT pass scrutiny: BenchmarkNarrowness produces a FATAL finding"
        )

    def test_report_has_fatal_and_critical(self):
        report = PipelineCritic().run()
        assert report.fatal_count >= 1
        assert report.critical_count >= 1

    def test_findings_sorted_by_severity(self):
        """FATAL findings must come before CRITICAL, CRITICAL before WARNING."""
        order = {
            FindingSeverity.FATAL: 0,
            FindingSeverity.CRITICAL: 1,
            FindingSeverity.WARNING: 2,
            FindingSeverity.INFO: 3,
        }
        report = PipelineCritic().run()
        severities = [order[f.severity] for f in report.findings]
        assert severities == sorted(severities), "Findings must be sorted by severity (FATAL first)"

    def test_all_six_categories_represented(self):
        """Every audit class must contribute at least one finding."""
        report = PipelineCritic().run()
        cats = {f.category for f in report.findings}
        expected = {
            "statistical_validity",
            "multiple_testing",
            "experimental_design",
            "pretraining_bias",
            "comparison_fairness",
            "benchmark_validity",
        }
        missing = expected - cats
        assert not missing, f"Missing finding categories: {missing}"

    def test_to_dict_is_serialisable(self):
        """CritiqueReport.to_dict() must return a JSON-serialisable structure."""
        import json

        report = PipelineCritic().run()
        d = report.to_dict()
        # Should not raise
        json.dumps(d)

    def test_print_loud_does_not_crash(self, capsys):
        """print_loud() must not raise (even with exit_on_fatal=False)."""
        report = PipelineCritic().run()
        report.print_loud(exit_on_fatal=False)
        captured = capsys.readouterr()
        assert "PIPELINE CRITIC" in captured.out
        assert "VERDICT" in captured.out

    def test_print_loud_exit_on_fatal_calls_sys_exit(self):
        """print_loud(exit_on_fatal=True) must call sys.exit(1) when FATAL found."""
        report = PipelineCritic().run()
        assert not report.passed  # Precondition: report has a FATAL finding
        with pytest.raises(SystemExit) as exc_info:
            report.print_loud(exit_on_fatal=True)
        assert exc_info.value.code == 1
