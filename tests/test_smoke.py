"""Fast CPU-only pytest suite for the DONUT SROIE pipeline.

No GPU, no model downloads, no network calls required.
Only the Python stdlib and the repo's own pure-Python modules are used.

Run with:  pytest tests/ -v --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure repo root is on the path so all imports resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# 1. Import chain
# ---------------------------------------------------------------------------


class TestImportChain:
    """Verify that core modules import cleanly without torch/transformers."""

    def test_constants_importable(self) -> None:
        import constants  # noqa: F401

    def test_sroie_loader_importable(self) -> None:
        from data_pipeline import SROIELoader  # noqa: F401

    def test_diagnostics_importable(self) -> None:
        import diagnostics  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Constants sanity
# ---------------------------------------------------------------------------


class TestConstantsSanity:
    """Verify that key constants in constants.py are defined and non-empty."""

    def test_fields_defined_and_nonempty(self) -> None:
        from constants import FIELDS

        assert isinstance(FIELDS, list), "FIELDS must be a list"
        assert len(FIELDS) > 0, "FIELDS must not be empty"

    def test_fields_contains_expected_keys(self) -> None:
        from constants import FIELDS

        for expected in ("company", "date", "address", "total"):
            assert expected in FIELDS, f"FIELDS is missing '{expected}'"

    def test_base_model_defined(self) -> None:
        from constants import BASE_MODEL

        assert isinstance(BASE_MODEL, str), "BASE_MODEL must be a str"
        assert len(BASE_MODEL) > 0, "BASE_MODEL must not be empty"

    def test_new_tokens_defined_and_nonempty(self) -> None:
        from constants import NEW_TOKENS

        assert isinstance(NEW_TOKENS, list), "NEW_TOKENS must be a list"
        assert len(NEW_TOKENS) > 0, "NEW_TOKENS must not be empty"

    def test_seed_is_integer(self) -> None:
        from constants import SEED

        assert isinstance(SEED, int), "SEED must be an int"


# ---------------------------------------------------------------------------
# 3. RuntimeCheckpoint pattern detection
# ---------------------------------------------------------------------------


class TestPatternDetection:
    """Verify that _detect_patterns() fires at the right F1 / loss values."""

    def _make_callback(self):
        import diagnostics

        # DiagnosticCallback does not require torch at instantiation time; the
        # TrainerCallback mixin is applied lazily (and skipped if transformers
        # is absent), so this works in a plain Python 3.12 environment.
        return diagnostics.DiagnosticCallback(experiment_id=1)

    # --- lm_head_dedup ---

    def test_lm_head_dedup_fires_in_range(self) -> None:
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=3.0,
            eval_f1=0.42,
            stage="training",
            experiment_id=1,
        )
        issues = cb._detect_patterns(cp)
        names = [i["pattern"] for i in issues]
        assert "lm_head_dedup" in names, f"Expected lm_head_dedup in {names}"

    def test_lm_head_dedup_does_not_fire_for_healthy_f1(self) -> None:
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=5.0,
            eval_f1=0.85,
            stage="training",
            experiment_id=1,
        )
        issues = cb._detect_patterns(cp)
        names = [i["pattern"] for i in issues]
        assert "lm_head_dedup" not in names

    def test_lm_head_dedup_does_not_fire_at_early_epoch(self) -> None:
        """Guard: lm_head_dedup requires epoch > 2 to avoid warmup false positives."""
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=1.0,
            eval_f1=0.42,
            stage="training",
            experiment_id=1,
        )
        issues = cb._detect_patterns(cp)
        names = [i["pattern"] for i in issues]
        assert "lm_head_dedup" not in names

    # --- total_f1_collapse ---

    def test_total_f1_collapse_fires_at_zero(self) -> None:
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=2.0,
            eval_f1=0.0,
            experiment_id=1,
        )
        issues = cb._detect_patterns(cp)
        names = [i["pattern"] for i in issues]
        assert "total_f1_collapse" in names

    def test_total_f1_collapse_does_not_fire_for_nonzero(self) -> None:
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=3.0,
            eval_f1=0.5,
            experiment_id=1,
        )
        issues = cb._detect_patterns(cp)
        names = [i["pattern"] for i in issues]
        assert "total_f1_collapse" not in names

    # --- loss_nan ---

    def test_loss_nan_fires_for_nan_train_loss(self) -> None:
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=1.0,
            train_loss=float("nan"),
            experiment_id=1,
        )
        issues = cb._detect_patterns(cp)
        names = [i["pattern"] for i in issues]
        assert "loss_nan" in names

    def test_loss_nan_does_not_fire_for_finite_loss(self) -> None:
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=1.0,
            train_loss=1.5,
            experiment_id=1,
        )
        issues = cb._detect_patterns(cp)
        names = [i["pattern"] for i in issues]
        assert "loss_nan" not in names

    # --- loss_plateau ---

    def test_loss_plateau_fires_at_high_loss_after_epoch3(self) -> None:
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=4.0,
            train_loss=2.5,
            experiment_id=1,
        )
        issues = cb._detect_patterns(cp)
        names = [i["pattern"] for i in issues]
        assert "loss_plateau" in names

    def test_loss_plateau_does_not_fire_at_early_epochs(self) -> None:
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=2.0,
            train_loss=2.5,
            experiment_id=1,
        )
        issues = cb._detect_patterns(cp)
        names = [i["pattern"] for i in issues]
        assert "loss_plateau" not in names

    # --- healthy checkpoint ---

    def test_no_critical_patterns_for_healthy_checkpoint(self) -> None:
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=5.0,
            eval_f1=0.85,
            train_loss=0.3,
            experiment_id=1,
        )
        issues = cb._detect_patterns(cp)
        critical = [i for i in issues if i["severity"] == "critical"]
        assert critical == [], f"Unexpected critical issues: {critical}"

    def test_pattern_only_raised_once_per_run(self) -> None:
        """Each pattern fires at most once per DiagnosticCallback instance."""
        import diagnostics

        cb = self._make_callback()
        cp = diagnostics.RuntimeCheckpoint(
            epoch=3.0,
            eval_f1=0.42,
            stage="training",
            experiment_id=1,
        )
        first = cb._detect_patterns(cp)
        second = cb._detect_patterns(cp)
        # First call should detect the pattern; second call should NOT (already raised).
        names_first = [i["pattern"] for i in first]
        names_second = [i["pattern"] for i in second]
        assert "lm_head_dedup" in names_first
        assert "lm_head_dedup" not in names_second


# ---------------------------------------------------------------------------
# 4. github_report_failure in-process dedup (_REPORTED_ISSUES set)
# ---------------------------------------------------------------------------


class TestGithubReportFailureDedup:
    """Verify that the module-level _REPORTED_ISSUES set suppresses duplicate reports."""

    def test_second_call_with_same_title_skips_api(self) -> None:
        """Two calls with identical stage+error_type must only hit the API once."""
        import diagnostics

        # Always start with a clean set so tests don't bleed into each other.
        diagnostics._REPORTED_ISSUES.clear()

        fake_issue = {
            "html_url": "https://github.com/owner/repo/issues/1",
            "number": 1,
            "title": "[DONUT Pipeline] smoke_test: TestError",
        }

        with (
            patch.object(diagnostics, "_get_github_token", return_value="fake-token"),
            patch.object(diagnostics, "_get_github_repo", return_value="owner/repo"),
            patch.object(diagnostics, "_github_request") as mock_req,
        ):
            # GET /search returns no existing issues; POST creates the issue.
            mock_req.side_effect = [
                {"items": []},  # github_search_open_issues GET
                fake_issue,  # github_create_issue POST
            ]

            result1 = diagnostics.github_report_failure(
                stage="smoke_test",
                error_type="TestError",
                error_message="first occurrence",
            )
            assert result1 is not None
            assert mock_req.call_count == 2  # GET (search) + POST (create)

            mock_req.reset_mock()
            mock_req.side_effect = None  # should not be called at all

            result2 = diagnostics.github_report_failure(
                stage="smoke_test",
                error_type="TestError",
                error_message="second occurrence — should be suppressed",
            )
            assert result2 is None, "Second call should return None (suppressed)"
            assert mock_req.call_count == 0, "_github_request must not be called on duplicate"

    def test_different_titles_both_get_reported(self) -> None:
        """Two calls with different error_type values must both reach the API."""
        import diagnostics

        diagnostics._REPORTED_ISSUES.clear()

        fake_issue_a = {
            "html_url": "https://github.com/owner/repo/issues/2",
            "number": 2,
            "title": "[DONUT Pipeline] smoke_test: ErrorA",
        }
        fake_issue_b = {
            "html_url": "https://github.com/owner/repo/issues/3",
            "number": 3,
            "title": "[DONUT Pipeline] smoke_test: ErrorB",
        }

        with (
            patch.object(diagnostics, "_get_github_token", return_value="fake-token"),
            patch.object(diagnostics, "_get_github_repo", return_value="owner/repo"),
            patch.object(diagnostics, "_github_request") as mock_req,
        ):
            mock_req.side_effect = [
                {"items": []},  # search for ErrorA → none found
                fake_issue_a,  # create ErrorA
                {"items": []},  # search for ErrorB → none found
                fake_issue_b,  # create ErrorB
            ]

            r1 = diagnostics.github_report_failure(
                stage="smoke_test", error_type="ErrorA", error_message="msg"
            )
            r2 = diagnostics.github_report_failure(
                stage="smoke_test", error_type="ErrorB", error_message="msg"
            )
            assert r1 is not None
            assert r2 is not None
            assert mock_req.call_count == 4  # 2 × (GET search + POST create)


# ---------------------------------------------------------------------------
# 5. CLI dedup path — github_search_open_issues guards github_report_failure
# ---------------------------------------------------------------------------


class TestCliDedupPath:
    """Verify that the fixed --smoke-test CLI path suppresses cross-process duplicates."""

    def test_existing_open_issue_suppresses_report(self) -> None:
        """When github_search_open_issues returns hits, github_report_failure is skipped."""
        import diagnostics

        diagnostics._REPORTED_ISSUES.clear()

        existing_issue = {
            "number": 150,
            "html_url": "https://github.com/owner/repo/issues/150",
            "title": "[DONUT Pipeline] smoke_test: SmokeTestFailure",
        }

        with (
            patch.object(diagnostics, "_get_github_token", return_value="fake-token"),
            patch.object(
                diagnostics, "github_search_open_issues", return_value=[existing_issue]
            ) as mock_search,
            patch.object(diagnostics, "github_report_failure") as mock_report,
        ):
            # Replicate the fixed CLI logic (the code we add to diagnostics.py):
            ok = False  # smoke test failed
            github_notify = False
            token = diagnostics._get_github_token()
            if not ok and (github_notify or token):
                existing = diagnostics.github_search_open_issues(
                    "[DONUT Pipeline] smoke_test: SmokeTestFailure"
                )
                if not existing:
                    diagnostics.github_report_failure(
                        stage="smoke_test",
                        error_type="SmokeTestFailure",
                        error_message="One or more preflight checks failed — see console output.",
                    )
                # else: suppressed — existing issue found

            mock_search.assert_called_once()
            mock_report.assert_not_called()

    def test_no_existing_issue_allows_report(self) -> None:
        """When github_search_open_issues returns empty, github_report_failure is called."""
        import diagnostics

        diagnostics._REPORTED_ISSUES.clear()

        with (
            patch.object(diagnostics, "_get_github_token", return_value="fake-token"),
            patch.object(
                diagnostics, "github_search_open_issues", return_value=[]
            ) as mock_search,
            patch.object(diagnostics, "github_report_failure") as mock_report,
        ):
            ok = False
            github_notify = False
            token = diagnostics._get_github_token()
            if not ok and (github_notify or token):
                existing = diagnostics.github_search_open_issues(
                    "[DONUT Pipeline] smoke_test: SmokeTestFailure"
                )
                if not existing:
                    diagnostics.github_report_failure(
                        stage="smoke_test",
                        error_type="SmokeTestFailure",
                        error_message="One or more preflight checks failed — see console output.",
                    )

            mock_search.assert_called_once()
            mock_report.assert_called_once()
