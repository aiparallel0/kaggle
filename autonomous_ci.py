# =============================================================================
# autonomous_ci.py
# Purpose: Fully autonomous CI/CD pipeline with AI evaluation and auto-merge.
#
# Pipeline stages:
#   1. Detect current branch and get PR number
#   2. Run smoke test + linting
#   3. Evaluate results with Claude/Mistral AI
#   4. Post results comment to GitHub PR
#   5. Auto-merge when all checks pass
#
# Usage:
#   python autonomous_ci.py                          # auto-detect branch & PR
#   python autonomous_ci.py --branch main            # manual override
#   python autonomous_ci.py --no-merge               # test without merging
#   python autonomous_ci.py --pr-only                # just post PR comment (no merge)
# =============================================================================

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# ============================================================================
# 1. Test Suite Runner
# ============================================================================


@dataclass
class TestResult:
    """Single test result with pass/fail and details."""

    name: str
    passed: bool
    duration_sec: float
    output: str  # stdout/stderr
    error: str | None = None  # exception message if failed


@dataclass
class TestSuiteResult:
    """Aggregate result from running full test suite."""

    started_at: float
    completed_at: float
    tests: list[TestResult]
    summary: str

    @property
    def total_tests(self) -> int:
        return len(self.tests)

    @property
    def passed_tests(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed_tests(self) -> int:
        return self.total_tests - self.passed_tests

    @property
    def all_passed(self) -> bool:
        return self.failed_tests == 0

    @property
    def duration_sec(self) -> float:
        return self.completed_at - self.started_at

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_sec": round(self.duration_sec, 2),
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "all_passed": self.all_passed,
            "summary": self.summary,
            "tests": [
                {
                    "name": t.name,
                    "passed": t.passed,
                    "duration_sec": round(t.duration_sec, 2),
                    "error": t.error,
                    "output": t.output[:500],  # truncate long output
                }
                for t in self.tests
            ],
        }


def run_smoke_test() -> TestResult:
    """Run diagnostics smoke test."""
    t0 = time.time()
    try:
        result = subprocess.run(
            ["python", "diagnostics.py", "--smoke-test"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        passed = result.returncode == 0
        output = result.stdout + "\n" + result.stderr
        return TestResult(
            name="smoke_test",
            passed=passed,
            duration_sec=time.time() - t0,
            output=output,
            error=None if passed else f"Exit code {result.returncode}",
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            name="smoke_test",
            passed=False,
            duration_sec=time.time() - t0,
            output="",
            error="Timeout after 60s",
        )
    except Exception as exc:
        return TestResult(
            name="smoke_test",
            passed=False,
            duration_sec=time.time() - t0,
            output="",
            error=str(exc),
        )


def run_ruff_lint() -> TestResult:
    """Run ruff linting."""
    t0 = time.time()
    try:
        result = subprocess.run(
            ["ruff", "check", "."],
            capture_output=True,
            text=True,
            timeout=30,
        )
        passed = result.returncode == 0
        output = result.stdout + "\n" + result.stderr
        return TestResult(
            name="ruff_lint",
            passed=passed,
            duration_sec=time.time() - t0,
            output=output,
            error=None if passed else "Linting failed",
        )
    except FileNotFoundError:
        return TestResult(
            name="ruff_lint",
            passed=False,
            duration_sec=time.time() - t0,
            output="",
            error="ruff not installed",
        )
    except Exception as exc:
        return TestResult(
            name="ruff_lint",
            passed=False,
            duration_sec=time.time() - t0,
            output="",
            error=str(exc),
        )


def run_import_check() -> TestResult:
    """Check that critical imports work."""
    t0 = time.time()
    try:
        result = subprocess.run(
            [
                "python",
                "-c",
                (
                    "from constants import FIELDS, BASE_MODEL, SEED; "
                    "from data_pipeline import SROIELoader; "
                    "from diagnostics import ai_diagnose, github_create_issue"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        passed = result.returncode == 0
        output = result.stdout + "\n" + result.stderr
        return TestResult(
            name="import_check",
            passed=passed,
            duration_sec=time.time() - t0,
            output=output,
            error=None if passed else "Import failed",
        )
    except Exception as exc:
        return TestResult(
            name="import_check",
            passed=False,
            duration_sec=time.time() - t0,
            output="",
            error=str(exc),
        )


def run_test_suite() -> TestSuiteResult:
    """Execute all tests and return aggregate result."""
    t0 = time.time()
    logger.info("[CI] Starting test suite...")

    tests = [
        run_import_check(),
        run_ruff_lint(),
        run_smoke_test(),
    ]

    passed = sum(1 for t in tests if t.passed)
    total = len(tests)
    summary = f"{passed}/{total} tests passed"

    result = TestSuiteResult(
        started_at=t0,
        completed_at=time.time(),
        tests=tests,
        summary=summary,
    )

    logger.info(f"[CI] Test suite completed: {summary}")
    for test in tests:
        status = "✓ PASS" if test.passed else "✗ FAIL"
        logger.info(f"    {status} — {test.name} ({test.duration_sec:.1f}s)")

    return result


# ============================================================================
# 2. AI Evaluation
# ============================================================================

_AI_EVALUATOR_PROMPT = """\
You are a CI/CD evaluation agent for a DONUT SROIE receipt extraction pipeline.
Analyze the test results below and provide:

1. **Pass/Fail verdict**: Is the build safe to merge?
2. **Root cause** (if failed): What broke?
3. **Recommended action**:
   - "MERGE" — all checks pass, safe to merge to main
   - "BLOCK" — critical failure, do not merge
   - "COMMENT_ONLY" — warning/issue found, post comment but don't block

Keep response under 300 words. Be concise."""


def evaluate_test_results(
    test_result: TestSuiteResult,
    provider: str = "auto",
    claude_model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Send test results to AI for evaluation.

    Returns dict with keys: verdict (MERGE/BLOCK/COMMENT_ONLY), reasoning, issues.
    """
    from diagnostics import ai_diagnose

    context = {
        "test_results": test_result.to_dict(),
        "evaluation_task": "Determine if build is safe to merge",
    }

    evaluation_text = ai_diagnose(
        context,
        provider=provider,
        claude_model=claude_model,
    )

    if not evaluation_text:
        # No AI available — default to conservative verdict
        verdict = "BLOCK" if test_result.failed_tests > 0 else "MERGE"
        reasoning = (
            f"AI evaluation unavailable; tests show {test_result.summary}. Defaulting to {verdict}."
        )
        return {
            "verdict": verdict,
            "reasoning": reasoning,
            "ai_available": False,
            "all_passed": test_result.all_passed,
        }

    # Parse AI response for verdict
    verdict = "BLOCK"  # conservative default
    if "MERGE" in evaluation_text.upper():
        verdict = "MERGE"
    elif "COMMENT" in evaluation_text.upper():
        verdict = "COMMENT_ONLY"

    return {
        "verdict": verdict,
        "reasoning": evaluation_text,
        "ai_available": True,
        "all_passed": test_result.all_passed,
    }


# ============================================================================
# 3. GitHub Integration
# ============================================================================


def get_current_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception as exc:
        logger.error(f"Failed to get current branch: {exc}")
        return ""


def get_pr_number_for_branch(branch: str, repo: str | None = None) -> int | None:
    """Get PR number for a branch using GitHub API."""
    from diagnostics import _get_github_repo, _get_github_token, _github_request

    token = _get_github_token()
    repo = repo or _get_github_repo()

    if not token or not repo:
        logger.debug("No GitHub credentials; skipping PR lookup")
        return None

    if not branch or branch == "HEAD" or branch in ["main", "master"]:
        logger.debug(f"Branch {branch!r} is not a feature branch; skipping PR lookup")
        return None

    result = _github_request(
        "GET",
        f"/repos/{repo}/pulls?head={repo.split('/')[0]}:{branch}&state=open",
        token,
    )

    if isinstance(result, list) and len(result) > 0:
        return result[0]["number"]
    return None


def post_ci_results_comment(
    test_result: TestSuiteResult,
    evaluation: dict,
    issue_number: int,
    repo: str | None = None,
) -> dict | None:
    """Post CI test results as a GitHub comment on a PR."""
    from diagnostics import github_post_comment

    # Build comment body
    lines = [
        "## Autonomous CI/CD Report",
        "",
        f"**Status:** {'✅ PASS' if test_result.all_passed else '❌ FAIL'}",
        f"**Duration:** {test_result.duration_sec:.1f}s",
        "",
    ]

    # Test results table
    lines += [
        "### Test Results",
        "",
        "| Test | Status | Duration |",
        "|---|---|---|",
    ]
    for test in test_result.tests:
        status = "✅" if test.passed else "❌"
        lines.append(f"| {test.name} | {status} | {test.duration_sec:.1f}s |")

    lines += ["", "### AI Evaluation", ""]
    if evaluation.get("ai_available"):
        lines.append(f"**Verdict:** `{evaluation['verdict']}`")
        lines.append("")
        lines.append("**Reasoning:**")
        lines.append(f"> {evaluation['reasoning'][:500]}")
    else:
        lines.append("*(AI evaluation unavailable)*")

    lines += [
        "",
        "### Recommended Action",
        f"- Verdict: **{evaluation['verdict']}**",
    ]

    if evaluation["verdict"] == "MERGE":
        lines.append("- Action: Auto-merge to main when all checks pass ✅")
    elif evaluation["verdict"] == "BLOCK":
        lines.append("- Action: Manual review required before merge 🔴")
    else:
        lines.append("- Action: Review feedback and update PR")

    body = "\n".join(lines)
    return github_post_comment(issue_number, body, repo=repo)


def auto_merge_pr(issue_number: int, repo: str | None = None) -> bool:
    """Merge PR when all checks pass."""
    from diagnostics import _get_github_repo, _get_github_token, _github_request

    token = _get_github_token()
    repo = repo or _get_github_repo()

    if not token or not repo:
        logger.debug("No GitHub credentials; skipping auto-merge")
        return False

    logger.info(f"[CI] Auto-merging PR #{issue_number}...")

    result = _github_request(
        "PUT",
        f"/repos/{repo}/pulls/{issue_number}/merge",
        token,
        {
            "commit_title": f"Merge PR #{issue_number} (auto-merged by CI)",
            "commit_message": "Automatically merged by autonomous CI/CD pipeline after passing all tests.",
            "merge_method": "squash",
        },
    )

    if result and result.get("merged"):
        logger.info(f"[CI] PR #{issue_number} merged successfully")
        print(f"✅ PR #{issue_number} merged to main")
        return True

    if result and result.get("message"):
        logger.warning(f"[CI] Merge failed: {result['message']}")
        print(f"⚠️  Could not merge PR: {result['message']}")
    return False


# ============================================================================
# 4. Main Orchestrator
# ============================================================================


def autonomous_pipeline(
    branch: str | None = None,
    pr_number: int | None = None,
    no_merge: bool = False,
    pr_only: bool = False,
    ai_provider: str = "auto",
    repo: str | None = None,
) -> bool:
    """Run the full autonomous CI/CD pipeline.

    Parameters
    ----------
    branch : str or None
        Git branch name. Auto-detected if None.
    pr_number : int or None
        GitHub PR number. Auto-detected from branch if None.
    no_merge : bool
        If True, don't auto-merge even if tests pass.
    pr_only : bool
        If True, only post PR comment; don't merge.
    ai_provider : str
        AI provider for evaluation ("claude", "mistral", or "auto").
    repo : str or None
        GitHub repo slug (e.g., "owner/repo"). Falls back to GITHUB_REPO env var.

    Returns
    -------
    bool
        True if all tests pass and (if attempted) merge succeeded.
    """
    from diagnostics import _get_github_repo

    # 1. Detect branch and PR
    if branch is None:
        branch = get_current_branch()
    if not branch:
        logger.error("[CI] Could not detect current branch")
        return False

    logger.info(f"[CI] Branch: {branch}")

    if pr_number is None:
        pr_number = get_pr_number_for_branch(branch, repo=repo)

    repo = repo or _get_github_repo()
    logger.info(f"[CI] PR: #{pr_number}" if pr_number else "[CI] Not a PR branch")

    # 2. Run tests
    test_result = run_test_suite()

    # 3. AI evaluation
    evaluation = evaluate_test_results(test_result, provider=ai_provider)
    logger.info(f"[CI] AI verdict: {evaluation['verdict']}")

    # 4. Post to GitHub (if PR exists)
    if pr_number and repo:
        post_ci_results_comment(test_result, evaluation, pr_number, repo=repo)

    # 5. Auto-merge (if verdict is MERGE and conditions met)
    if evaluation["verdict"] == "MERGE" and pr_number and repo and not no_merge and not pr_only:
        success = auto_merge_pr(pr_number, repo=repo)
        return success and test_result.all_passed

    # If pr_only or no_merge, just return test status
    return test_result.all_passed


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous CI/CD pipeline with AI evaluation")
    parser.add_argument(
        "--branch",
        default=None,
        help="Git branch (auto-detected if not specified)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        metavar="N",
        help="PR number (auto-detected if not specified)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        metavar="OWNER/REPO",
        help="GitHub repo slug (default: GITHUB_REPO env var)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Run tests but don't auto-merge",
    )
    parser.add_argument(
        "--pr-only",
        action="store_true",
        help="Post PR comment only; don't merge",
    )
    parser.add_argument(
        "--ai-provider",
        choices=["claude", "mistral", "auto"],
        default="auto",
        help="AI provider for evaluation",
    )

    args = parser.parse_args()

    success = autonomous_pipeline(
        branch=args.branch,
        pr_number=args.pr,
        no_merge=args.no_merge,
        pr_only=args.pr_only,
        ai_provider=args.ai_provider,
        repo=args.repo,
    )

    sys.exit(0 if success else 1)
