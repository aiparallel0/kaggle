# =============================================================================
# autonomous_ci.py
# Purpose: Fully autonomous CI/CD pipeline with AI evaluation, auto-fix, and auto-merge.
#
# Pipeline stages:
#   1. Detect current branch and get PR number
#   2. Run smoke test + linting
#   3. Evaluate results with Claude/Mistral AI
#   4. [NEW] If BLOCK verdict: Auto-generate fixes and retry (up to N times)
#   5. Post results comment to GitHub PR
#   6. Auto-merge when all checks pass
#
# Usage:
#   python autonomous_ci.py                          # auto-detect branch & PR
#   python autonomous_ci.py --branch main            # manual override
#   python autonomous_ci.py --no-merge               # test without merging
#   python autonomous_ci.py --max-fix-attempts 5     # max 5 auto-fix retries
#   python autonomous_ci.py --no-auto-fix            # disable auto-fix (test-only)
# =============================================================================

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# Files that auto-fix is allowed to modify
AUTOFIX_WHITELIST = {
    "constants.py",
    "train.py",
    "data_pipeline.py",
    "run_experiments.py",
    "run_all.py",
    "diagnostics.py",
    "autonomous_ci.py",
}

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
    fix_attempts: list | None = None,
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

    # Auto-fix attempts (if any)
    if fix_attempts:
        lines += ["### Auto-Fix Attempts", ""]
        for attempt in fix_attempts:
            status = "✅" if attempt.success else "❌"
            lines.append(f"**Attempt {attempt.attempt_num}/{attempt.max_attempts}:** {status}")
            if attempt.fix_description:
                lines.append(f"- Fix: {attempt.fix_description}")
            if attempt.error:
                lines.append(f"- Error: {attempt.error[:100]}")
        lines += [""]

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
    max_fix_attempts: int = 10,
    no_auto_fix: bool = False,
) -> bool:
    """Run the full autonomous CI/CD pipeline with auto-fix loop.

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
    max_fix_attempts : int
        Maximum number of auto-fix attempts (default 10).
    no_auto_fix : bool
        If True, disable auto-fix; only run tests.

    Returns
    -------
    bool
        True if all tests pass and (if attempted) merge succeeded.
    """
    from diagnostics import _get_github_repo

    logger.info(f"[CI] Auto-fix: {not no_auto_fix} (max {max_fix_attempts} attempts)")

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

    # 4. [NEW] Auto-fix loop if tests failed
    fix_attempts = []
    if evaluation["verdict"] == "BLOCK" and not no_auto_fix:
        logger.warning("[CI] Tests failed. Starting auto-fix loop...")
        success, fix_attempts = auto_fix_and_retry(
            test_result, max_attempts=max_fix_attempts, provider=ai_provider
        )

        if success:
            logger.info("[CI] ✅ Auto-fix succeeded! Re-evaluating...")
            test_result = run_test_suite()
            evaluation = evaluate_test_results(test_result, provider=ai_provider)
        else:
            logger.error(f"[CI] ❌ Auto-fix failed after {max_fix_attempts} attempts")

    # 5. Post to GitHub (if PR exists)
    if pr_number and repo:
        post_ci_results_comment(
            test_result, evaluation, pr_number, repo=repo, fix_attempts=fix_attempts
        )

    # 6. Auto-merge (if verdict is MERGE and conditions met)
    if evaluation["verdict"] == "MERGE" and pr_number and repo and not no_merge and not pr_only:
        success = auto_merge_pr(pr_number, repo=repo)
        return success and test_result.all_passed

    # If pr_only or no_merge, just return test status
    return test_result.all_passed


# ============================================================================
# 4. Autonomous Auto-Fix Loop
# ============================================================================

_REPO_ROOT = Path(__file__).parent

_AUTOFIX_SYSTEM_PROMPT = (
    "You are an expert Python code fixer for a machine learning pipeline codebase. "
    "You will be given the CURRENT CONTENT of a Python file together with the exact error "
    "that the test suite reported for it. "
    "Return ONLY the complete, corrected Python file content — no markdown fences, "
    "no ``` delimiters, no explanations, no commentary. "
    "Just the raw Python source code that should replace the file, starting with the "
    "first line of the file (imports / module docstring) and ending with the last line."
)


@dataclass
class AutoFixAttempt:
    """Result of a single auto-fix attempt."""

    attempt_num: int
    max_attempts: int
    test_result: TestSuiteResult | None = None
    fix_applied: bool = False
    fix_description: str = ""
    error: str | None = None
    success: bool = False


def _validate_python_syntax(code: str) -> tuple[bool, str | None]:
    """Validate Python code syntax without executing it.

    Returns (is_valid, error_message).
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        return False, f"ParseError: {e}"


def _strip_markdown_fences(text: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` wrappers that AI sometimes adds."""
    text = text.strip()
    # Remove leading fence (```python or ```)
    text = re.sub(r"^```(?:python)?\s*\n?", "", text)
    # Remove trailing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _identify_files_to_fix(test_result: TestSuiteResult) -> list[tuple[str, str]]:
    """Map failing tests to candidate files that need fixing.

    Returns a deduplicated list of (filename, combined_error_context) pairs,
    ordered from most-specific to least-specific guess.
    """
    candidates: list[tuple[str, str]] = []  # (filename, error_context)

    for test in test_result.tests:
        if test.passed:
            continue
        output = (test.output or "") + "\n" + (test.error or "")

        if test.name == "import_check":
            # The error output names the module that failed.
            # e.g. "ImportError: cannot import name 'X' from 'constants'"
            # Map module names to filenames.
            module_to_file = {
                "constants": "constants.py",
                "data_pipeline": "data_pipeline.py",
                "diagnostics": "diagnostics.py",
                "train": "train.py",
                "run_experiments": "run_experiments.py",
                "run_all": "run_all.py",
                "autonomous_ci": "autonomous_ci.py",
            }
            matched = False
            for module, fname in module_to_file.items():
                if module in output:
                    candidates.append((fname, output))
                    matched = True
            if not matched:
                # Can't tell — try the three most-imported files
                for fname in ["constants.py", "data_pipeline.py", "diagnostics.py"]:
                    candidates.append((fname, output))

        elif test.name == "ruff_lint":
            # ruff output lines: "path/to/file.py:line:col: EXX message"
            for line in output.splitlines():
                m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_/]*\.py):\d+:\d+:", line)
                if m:
                    fname = Path(m.group(1)).name  # strip any leading path
                    if fname in AUTOFIX_WHITELIST:
                        candidates.append((fname, line))

        elif test.name == "smoke_test":
            # Smoke test errors are usually in diagnostics.py or constants.py
            for fname in ["diagnostics.py", "constants.py", "run_experiments.py"]:
                candidates.append((fname, output))

    # Deduplicate by filename; combine all error contexts for the same file
    seen: dict[str, list[str]] = {}
    for fname, ctx in candidates:
        seen.setdefault(fname, []).append(ctx)

    return [(fname, "\n---\n".join(ctxs)) for fname, ctxs in seen.items()]


def _generate_file_fix(
    filename: str,
    error_context: str,
    previous_failures: list[str],
    provider: str = "auto",
) -> str | None:
    """Ask AI to return a corrected version of *filename*.

    Returns the complete new file content as a string, or None if AI is unavailable.
    """
    from diagnostics import ai_diagnose

    full_path = _REPO_ROOT / filename
    try:
        current_content = full_path.read_text()
    except OSError as exc:
        logger.warning(f"[AutoFix] Cannot read {filename}: {exc}")
        return None

    prev_note = ""
    if previous_failures:
        prev_note = "\n\nPrevious fix attempts for this file failed:\n" + "\n".join(
            previous_failures[-3:]
        )

    context = {
        "filename": filename,
        "current_file_content": current_content,
        "errors": error_context[-3000:],
        "task": (
            f"The file '{filename}' is causing the test failures shown in 'errors'. "
            f"Return the COMPLETE corrected content of '{filename}'. "
            f"No markdown, no explanation — the raw Python code only."
        )
        + prev_note,
    }

    return ai_diagnose(
        context,
        provider=provider,
        system_prompt=_AUTOFIX_SYSTEM_PROMPT,
        max_tokens=4096,
    )


def auto_fix_and_retry(
    test_result: TestSuiteResult,
    max_attempts: int = 10,
    provider: str = "auto",
) -> tuple[bool, list[AutoFixAttempt]]:
    """Attempt to auto-fix test failures by having AI rewrite broken files.

    For each attempt:
      1. Identify which files are likely broken (from error output).
      2. Ask AI for the complete corrected file content.
      3. Validate syntax, strip markdown fences, write to disk.
      4. Run ruff --fix + ruff format to ensure style compliance.
      5. Git add + commit (skip gracefully if nothing changed).
      6. Re-run the full test suite.
      7. If all pass → return success.  Otherwise loop.

    Returns (success, list_of_attempts).
    """
    attempts: list[AutoFixAttempt] = []
    previous_failures: list[str] = []

    for attempt_num in range(1, max_attempts + 1):
        logger.info(f"[AutoFix] Attempt {attempt_num}/{max_attempts}")

        # 1. Identify candidate files
        files_to_fix = _identify_files_to_fix(test_result)
        if not files_to_fix:
            # Fallback: try the most fundamental file
            files_to_fix = [("constants.py", test_result.summary)]

        fix_applied = False
        fix_description_parts: list[str] = []
        fix_error: str | None = None
        patched_files: list[str] = []

        # 2. Generate + apply fix for each candidate file (cap at 3 per attempt)
        for filename, error_ctx in files_to_fix[:3]:
            logger.info(f"[AutoFix] Requesting AI fix for {filename}...")
            new_content = _generate_file_fix(
                filename, error_ctx, previous_failures, provider=provider
            )

            if not new_content:
                fix_error = f"AI returned no content for {filename}"
                logger.warning(f"[AutoFix] {fix_error}")
                continue

            # Strip markdown fences that AI sometimes wraps around code
            new_content = _strip_markdown_fences(new_content)

            # Validate syntax before writing
            is_valid, syntax_error = _validate_python_syntax(new_content)
            if not is_valid:
                fix_error = f"AI output for {filename} has syntax error: {syntax_error}"
                logger.warning(f"[AutoFix] {fix_error}")
                previous_failures.append(f"Attempt {attempt_num}: {fix_error}")
                continue

            # Write corrected content to disk
            full_path = _REPO_ROOT / filename
            try:
                full_path.write_text(new_content)
                logger.info(f"[AutoFix] Wrote fix to {filename}")
                fix_applied = True
                patched_files.append(filename)
                fix_description_parts.append(f"rewrote {filename}")
            except OSError as exc:
                fix_error = f"Could not write {filename}: {exc}"
                logger.error(f"[AutoFix] {fix_error}")
                continue

        # 3. Auto-format patched files with ruff (don't fail if ruff absent)
        if patched_files:
            subprocess.run(
                ["ruff", "check", "--fix"] + patched_files, capture_output=True, cwd=_REPO_ROOT
            )
            subprocess.run(["ruff", "format"] + patched_files, capture_output=True, cwd=_REPO_ROOT)

        # 4. Git add + commit (check return code — "nothing to commit" is OK)
        if patched_files:
            subprocess.run(
                ["git", "add", "--"] + patched_files, capture_output=True, cwd=_REPO_ROOT
            )
            commit_msg = (
                f"[auto-fix {attempt_num}/{max_attempts}] {'; '.join(fix_description_parts)}"
            )
            commit_result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=_REPO_ROOT,
            )
            combined_output = commit_result.stdout + commit_result.stderr
            if commit_result.returncode == 0:
                logger.info(f"[AutoFix] Committed: {commit_msg}")
            elif "nothing to commit" in combined_output:
                logger.info("[AutoFix] AI produced identical content — no commit needed")
            else:
                logger.warning(f"[AutoFix] git commit failed: {combined_output[:300]}")

        # 5. Re-run tests
        logger.info("[AutoFix] Re-running tests...")
        new_test_result = run_test_suite()

        attempt = AutoFixAttempt(
            attempt_num=attempt_num,
            max_attempts=max_attempts,
            test_result=new_test_result,
            fix_applied=fix_applied,
            fix_description=(
                "; ".join(fix_description_parts)
                if fix_description_parts
                else (fix_error or "No fix generated")
            ),
            error=None if new_test_result.all_passed else new_test_result.summary,
            success=new_test_result.all_passed,
        )
        attempts.append(attempt)

        if new_test_result.all_passed:
            logger.info(f"[AutoFix] Tests PASSED on attempt {attempt_num}!")
            return True, attempts

        # 6. Prepare next iteration
        test_result = new_test_result
        previous_failures.append(
            f"Attempt {attempt_num}: {test_result.summary} — fixed: {'; '.join(fix_description_parts) or 'nothing'}"
        )

    logger.error(f"[AutoFix] Failed to fix after {max_attempts} attempts")
    return False, attempts


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous CI/CD pipeline with AI evaluation and auto-fix"
    )
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
        help="AI provider for evaluation and auto-fix",
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=10,
        metavar="N",
        help="Maximum auto-fix retry attempts (default: 10)",
    )
    parser.add_argument(
        "--no-auto-fix",
        action="store_true",
        help="Disable auto-fix; only run tests and evaluate",
    )

    args = parser.parse_args()

    success = autonomous_pipeline(
        branch=args.branch,
        pr_number=args.pr,
        no_merge=args.no_merge,
        pr_only=args.pr_only,
        ai_provider=args.ai_provider,
        repo=args.repo,
        max_fix_attempts=args.max_fix_attempts,
        no_auto_fix=args.no_auto_fix,
    )

    sys.exit(0 if success else 1)
