"""Detector for the ~2-PR bugs documented in CLAUDE.md Section 5."""

import ast
import logging
import re
from pathlib import Path

from pipeline_types import BugPattern, BugReport, SeverityLevel

__all__ = ["BugPatternDetector"]

logger = logging.getLogger(__name__)


class BugPatternDetector:
    """Scan codebase for recurring bugs that silently break the pipeline.

    From CLAUDE.md Section 5 (The ~2-PR Bug Pattern):
    - Syntax errors (missing brackets/commas)
    - Python True/False/None vs JSON true/false/null
    - Missing tie_word_embeddings = False after resize_token_embeddings()
    - lm_head weight safetensors deduplication issues
    - token2json list output handling
    """

    # Patterns for Python/JSON confusion
    PYTHON_JSON_PATTERNS = [
        (r"\btrue\b(?!\w)", "Python uses True, not true"),
        (r"\bfalse\b(?!\w)", "Python uses False, not false"),
        (r"\bnull\b(?!\w)", "Python uses None, not null"),
    ]

    # Patterns for critical missing checks
    CRITICAL_PATTERNS = [
        (
            r"resize_token_embeddings\s*\(",
            "check_tie_word_embeddings",
            "Must set config.tie_word_embeddings = False after resize_token_embeddings()",
        ),
    ]

    @staticmethod
    def detect_syntax_errors(code: str, file_path: Path) -> list[BugPattern]:
        """Detect syntax errors (missing brackets, commas, etc).

        Args:
            code: Python source code
            file_path: Path to the file

        Returns:
            List of detected syntax bugs
        """
        bugs = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            bugs.append(
                BugPattern(
                    severity=SeverityLevel.CRITICAL,
                    category="syntax",
                    file=file_path,
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    description=f"Syntax error: {e.msg}",
                    code_snippet=e.text or "",
                    fix_suggestion=f"Fix syntax error at line {e.lineno}: {e.msg}",
                )
            )
        except Exception as e:
            bugs.append(
                BugPattern(
                    severity=SeverityLevel.WARNING,
                    category="syntax",
                    file=file_path,
                    line=0,
                    column=0,
                    description=f"Could not parse file: {str(e)}",
                    code_snippet="",
                    fix_suggestion="Manually review file for syntax errors",
                )
            )

        return bugs

    @staticmethod
    def detect_python_json_confusion(code: str, file_path: Path) -> list[BugPattern]:
        """Detect JSON literals in Python code (true/false/null instead of True/False/None).

        Args:
            code: Python source code
            file_path: Path to the file

        Returns:
            List of detected JSON confusion bugs
        """
        bugs = []
        lines = code.split("\n")

        for line_no, line in enumerate(lines, 1):
            # Skip comments and strings
            if line.strip().startswith("#"):
                continue
            if '"""' in line or "'''" in line:
                continue

            for pattern, _description in BugPatternDetector.PYTHON_JSON_PATTERNS:
                matches = re.finditer(pattern, line)
                for match in matches:
                    json_literal = match.group()
                    replacement = {
                        "true": "True",
                        "false": "False",
                        "null": "None",
                    }.get(json_literal, json_literal)

                    bugs.append(
                        BugPattern(
                            severity=SeverityLevel.CRITICAL,
                            category="compatibility",
                            file=file_path,
                            line=line_no,
                            column=match.start() + 1,
                            description=f"JSON literal '{json_literal}' in Python code",
                            code_snippet=line.strip(),
                            fix_suggestion=f"Replace '{json_literal}' with '{replacement}'",
                        )
                    )

        return bugs

    @staticmethod
    def detect_missing_tie_word_embeddings(code: str, file_path: Path) -> list[BugPattern]:
        """Detect missing config.tie_word_embeddings = False after resize_token_embeddings().

        This is critical - without this line, lm_head.weight gets dropped by safetensors,
        causing F1=0 on all predictions after checkpoint reload.

        From CLAUDE.md: "The single most destructive silent failure in the codebase."

        Args:
            code: Python source code
            file_path: Path to the file

        Returns:
            List of detected missing checks
        """
        bugs = []
        lines = code.split("\n")

        # Find resize_token_embeddings calls
        for line_no, line in enumerate(lines, 1):
            if "resize_token_embeddings" in line:
                # Check if tie_word_embeddings = False is set in next 5 lines
                found_tie_word = False
                for offset in range(1, min(6, len(lines) - line_no + 1)):
                    next_line = lines[line_no + offset - 1]
                    if "tie_word_embeddings" in next_line and "False" in next_line:
                        found_tie_word = True
                        break

                if not found_tie_word:
                    bugs.append(
                        BugPattern(
                            severity=SeverityLevel.CRITICAL,
                            category="logic",
                            file=file_path,
                            line=line_no,
                            column=1,
                            description="resize_token_embeddings() called without setting tie_word_embeddings=False",
                            code_snippet=line.strip(),
                            fix_suggestion=(
                                "Add this line after resize_token_embeddings():\n"
                                "model.config.tie_word_embeddings = False"
                            ),
                        )
                    )

        return bugs

    @staticmethod
    def detect_lm_head_issues(code: str, file_path: Path) -> list[BugPattern]:
        """Detect potential lm_head weight issues from safetensors deduplication.

        From CLAUDE.md BUG B: safetensors deduplication drops lm_head.weight

        Args:
            code: Python source code
            file_path: Path to the file

        Returns:
            List of detected lm_head issues
        """
        bugs = []

        # Check if LmHeadCloneCallback is implemented
        if "resize_token_embeddings" in code and "LmHeadCloneCallback" not in code:
            bugs.append(
                BugPattern(
                    severity=SeverityLevel.WARNING,
                    category="logic",
                    file=file_path,
                    line=0,
                    column=0,
                    description="File has resize_token_embeddings() but no LmHeadCloneCallback",
                    code_snippet="",
                    fix_suggestion=(
                        "Implement LmHeadCloneCallback to prevent lm_head weight deduplication"
                    ),
                )
            )

        return bugs

    @staticmethod
    def detect_token2json_list_handling(code: str, file_path: Path) -> list[BugPattern]:
        """Detect missing token2json list output handling.

        From CLAUDE.md BUG C: token2json can return list instead of dict when <sep/> is present.
        Must merge list into single dict.

        Args:
            code: Python source code
            file_path: Path to the file

        Returns:
            List of detected token2json issues
        """
        bugs = []

        if (
            "token2json" in code
            and "_parse_prediction" in code
            and "isinstance(result, list)" not in code
        ):
            bugs.append(
                BugPattern(
                    severity=SeverityLevel.WARNING,
                    category="logic",
                    file=file_path,
                    line=0,
                    column=0,
                    description="token2json list output handling not found",
                    code_snippet="",
                    fix_suggestion=(
                        "Add list handling in _parse_prediction():\n"
                        "if isinstance(result, list): merged = {}; "
                        "for page in result: merged.update(page)"
                    ),
                )
            )

        return bugs

    async def scan_codebase(self, root_dir: Path) -> BugReport:
        """Scan entire codebase for bugs.

        Args:
            root_dir: Root directory to scan

        Returns:
            Comprehensive bug report
        """
        report = BugReport()

        # Find all Python files
        python_files = list(root_dir.glob("**/*.py"))

        # Skip test directories and __pycache__
        python_files = [f for f in python_files if "__pycache__" not in str(f)]
        python_files = [f for f in python_files if "/.pytest_cache/" not in str(f)]

        logger.info(f"Scanning {len(python_files)} Python files for bugs...")

        for file_path in python_files:
            try:
                code = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                continue

            # Run all detectors
            all_bugs = []
            all_bugs.extend(self.detect_syntax_errors(code, file_path))
            all_bugs.extend(self.detect_python_json_confusion(code, file_path))
            all_bugs.extend(self.detect_missing_tie_word_embeddings(code, file_path))
            all_bugs.extend(self.detect_lm_head_issues(code, file_path))
            all_bugs.extend(self.detect_token2json_list_handling(code, file_path))

            # Add to report
            for bug in all_bugs:
                report.bugs.append(bug)
                if bug.severity == SeverityLevel.CRITICAL:
                    report.total_critical += 1
                elif bug.severity == SeverityLevel.WARNING:
                    report.total_warnings += 1

        logger.info(
            f"Scan complete: {report.total_critical} critical, {report.total_warnings} warnings"
        )

        return report

    async def scan_file(self, file_path: Path) -> list[BugPattern]:
        """Scan a single file for bugs.

        Args:
            file_path: Path to file to scan

        Returns:
            List of detected bugs
        """
        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return []

        bugs = []
        bugs.extend(self.detect_syntax_errors(code, file_path))
        bugs.extend(self.detect_python_json_confusion(code, file_path))
        bugs.extend(self.detect_missing_tie_word_embeddings(code, file_path))
        bugs.extend(self.detect_lm_head_issues(code, file_path))
        bugs.extend(self.detect_token2json_list_handling(code, file_path))

        return bugs
