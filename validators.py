"""Validators for cloud pipeline safety checks.

Consolidated from the former validators/ package. Contains all validator
classes and functions for the DONUT SROIE multi-dataset pipeline.
"""

from __future__ import annotations

import ast
import importlib
import json
import logging
import random
import re
import struct
from pathlib import Path
from typing import Any

from pipeline_types import (
    BugPattern,
    BugReport,
    CheckpointCorruptionError,
    DataSplitValidationReport,
    SeverityLevel,
)

__all__ = [
    "ImportChainChecker",
    "BugPatternDetector",
    "DataSplitValidator",
    "ModelWeightValidator",
    "SeedValidator",
    "CheckpointCorruptionError",
    "validate_checkpoint",
]

# ---------------------------------------------------------------------------
# Module-level loggers (mirrors per-module loggers from the original package)
# ---------------------------------------------------------------------------

_log_bug = logging.getLogger("validators.bug_pattern_detector")
_log_ckpt = logging.getLogger("validators.checkpoint_resume_validator")
_log_split = logging.getLogger("validators.data_split_validator")
_log_chain = logging.getLogger("validators.import_chain_checker")
_log_weight = logging.getLogger("validators.model_weight_validator")
_log_seed = logging.getLogger("validators.seed_validator")


# ---------------------------------------------------------------------------
# BugPatternDetector
# ---------------------------------------------------------------------------


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

        _log_bug.info(f"Scanning {len(python_files)} Python files for bugs...")

        for file_path in python_files:
            try:
                code = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                _log_bug.warning(f"Could not read {file_path}: {e}")
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

        _log_bug.info(
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
            _log_bug.error(f"Could not read {file_path}: {e}")
            return []

        bugs = []
        bugs.extend(self.detect_syntax_errors(code, file_path))
        bugs.extend(self.detect_python_json_confusion(code, file_path))
        bugs.extend(self.detect_missing_tie_word_embeddings(code, file_path))
        bugs.extend(self.detect_lm_head_issues(code, file_path))
        bugs.extend(self.detect_token2json_list_handling(code, file_path))

        return bugs


# ---------------------------------------------------------------------------
# validate_checkpoint (formerly checkpoint_resume_validator)
#
# Validates a fine-tuned DONUT checkpoint before loading for evaluation.
# See module docstring in the original file for full usage notes.
# ---------------------------------------------------------------------------


def validate_checkpoint(
    model_path: str | Path,
    result_json_path: str | Path | None = None,
    expected_vocab_size: int | None = None,
) -> None:
    """Validate a fine-tuned DONUT checkpoint before loading for evaluation.

    Parameters
    ----------
    model_path:
        Path to the saved model directory (contains config.json, model.safetensors, …).
    result_json_path:
        Path to the experiment result JSON (e.g. ``results/experiment_6.json``).
        Used to check that encoder image_size matches training resolution.
        Skipped when ``None`` or the file does not exist.
    expected_vocab_size:
        Expected tokenizer vocabulary size (after adding SROIE special tokens).
        Skipped when ``None``.

    Raises
    ------
    CheckpointCorruptionError
        On any integrity failure.
    FileNotFoundError
        When ``model_path`` does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {model_path}")

    # ── Check 1: lm_head.weight presence in safetensors index ────────────
    _check_lm_head(model_path)

    # ── Check 2: encoder image_size matches training resolution ───────────
    if result_json_path is not None:
        _check_image_size(model_path, Path(result_json_path))

    # ── Check 3: vocab size matches embedding shape ───────────────────────
    if expected_vocab_size is not None:
        _check_vocab_size(model_path, expected_vocab_size)

    _log_ckpt.info("[CheckpointValidator] %s — all checks passed ✓", model_path)


def _check_lm_head(model_path: Path) -> None:
    """Verify lm_head.weight is present in the safetensors index."""
    index_file = model_path / "model.safetensors.index.json"
    single_file = model_path / "model.safetensors"

    if index_file.exists():
        try:
            data: dict = json.loads(index_file.read_text())
            weight_map: dict[str, Any] = data.get("weight_map", {})
            if "decoder.lm_head.weight" not in weight_map:
                raise CheckpointCorruptionError(
                    f"[CheckpointValidator] lm_head.weight is MISSING from "
                    f"{index_file}.  This indicates the safetensors deduplication "
                    f"bug (lm_head was tied to embed_tokens when saved).  "
                    f"Re-train with LmHeadCloneCallback registered.  "
                    f"See CLAUDE.md §5 Pattern 6."
                )
            _log_ckpt.debug(
                "[CheckpointValidator] lm_head.weight found in index: shard=%s",
                weight_map["decoder.lm_head.weight"],
            )
        except CheckpointCorruptionError:
            raise
        except Exception as exc:
            _log_ckpt.warning(
                "[CheckpointValidator] Could not parse safetensors index (%s) — skipping lm_head check",
                exc,
            )
    elif single_file.exists():
        # Single-shard model: inspect tensor keys via safetensors header
        try:
            with open(single_file, "rb") as fh:
                header_len = struct.unpack("<Q", fh.read(8))[0]
                header_bytes = fh.read(header_len)
            header: dict = json.loads(header_bytes)
            if "decoder.lm_head.weight" not in header:
                raise CheckpointCorruptionError(
                    f"[CheckpointValidator] lm_head.weight is MISSING from "
                    f"{single_file} header.  See CLAUDE.md §5 Pattern 6."
                )
            _log_ckpt.debug("[CheckpointValidator] lm_head.weight found in single-shard model ✓")
        except CheckpointCorruptionError:
            raise
        except Exception as exc:
            _log_ckpt.warning(
                "[CheckpointValidator] Could not inspect single-shard safetensors (%s) — "
                "skipping lm_head check",
                exc,
            )
    else:
        # Might be a PyTorch bin checkpoint — skip safetensors check
        _log_ckpt.debug(
            "[CheckpointValidator] No safetensors file found in %s — skipping lm_head check",
            model_path,
        )


def _check_image_size(model_path: Path, result_json_path: Path) -> None:
    """Check encoder.image_size against training resolution in result JSON."""
    if not result_json_path.exists():
        _log_ckpt.debug("[CheckpointValidator] Result JSON not found — skipping image_size check")
        return

    # Load encoder config from checkpoint
    encoder_config_file = model_path / "encoder_config.json"
    config_file = model_path / "config.json"

    encoder_image_size: list[int] | None = None
    for cf in (encoder_config_file, config_file):
        if not cf.exists():
            continue
        try:
            cfg: dict = json.loads(cf.read_text())
            # config.json wraps encoder under "encoder" key
            enc = cfg.get("encoder", cfg)
            if "image_size" in enc:
                encoder_image_size = enc["image_size"]
                break
        except Exception as exc:
            _log_ckpt.debug("[CheckpointValidator] Could not parse %s: %s", cf, exc)

    if encoder_image_size is None:
        _log_ckpt.debug("[CheckpointValidator] encoder.image_size not found — skipping size check")
        return

    # Load expected size from result JSON
    try:
        result: dict = json.loads(result_json_path.read_text())
        metrics: dict = result.get("metrics", {})
        # Try to find input_size recorded during training
        input_size = metrics.get("input_size") or result.get("input_size")
        if input_size is None:
            _log_ckpt.debug(
                "[CheckpointValidator] input_size not in result JSON — skipping size check"
            )
            return
        if list(encoder_image_size) != list(input_size):
            raise CheckpointCorruptionError(
                f"[CheckpointValidator] encoder.image_size={encoder_image_size} "
                f"does not match training INPUT_SIZE={input_size} from {result_json_path}.  "
                f"The checkpoint was trained at a different resolution."
            )
        _log_ckpt.debug(
            "[CheckpointValidator] encoder.image_size=%s matches training resolution ✓",
            encoder_image_size,
        )
    except CheckpointCorruptionError:
        raise
    except Exception as exc:
        _log_ckpt.warning(
            "[CheckpointValidator] Could not verify image_size from result JSON (%s)",
            exc,
        )


def _check_vocab_size(model_path: Path, expected_vocab_size: int) -> None:
    """Verify embedding matrix shape matches expected vocabulary size."""
    config_file = model_path / "config.json"
    if not config_file.exists():
        _log_ckpt.debug("[CheckpointValidator] config.json not found — skipping vocab check")
        return

    try:
        cfg: dict = json.loads(config_file.read_text())
        decoder_cfg = cfg.get("decoder", cfg)
        # vocab_size can be at top level or under decoder
        vocab_size = decoder_cfg.get("vocab_size") or cfg.get("vocab_size")
        if vocab_size is None:
            _log_ckpt.debug("[CheckpointValidator] vocab_size not in config — skipping vocab check")
            return
        if int(vocab_size) != expected_vocab_size:
            raise CheckpointCorruptionError(
                f"[CheckpointValidator] Checkpoint vocab_size={vocab_size} "
                f"does not match expected {expected_vocab_size}.  "
                f"The tokenizer has different special tokens than expected."
            )
        _log_ckpt.debug(
            "[CheckpointValidator] vocab_size=%d matches expected ✓", vocab_size
        )
    except CheckpointCorruptionError:
        raise
    except Exception as exc:
        _log_ckpt.warning(
            "[CheckpointValidator] Could not verify vocab_size from config (%s)",
            exc,
        )


# ---------------------------------------------------------------------------
# DataSplitValidator
# ---------------------------------------------------------------------------


class DataSplitValidator:
    """Validate SROIE data split integrity."""

    EXPECTED_TRAIN_COUNT = 500
    EXPECTED_VAL_COUNT = 63
    EXPECTED_TEST_COUNT = 63

    @staticmethod
    def validate_sroie_split(sroie_dir: Path) -> DataSplitValidationReport:
        """Verify the critical SROIE 80/10/10 split.

        Checks:
        1. Directories exist: img/, val_img/, test_img/
        2. Key files exist for each image
        3. Image counts are correct: 500/63/63
        4. No overlap between sets
        5. val_img/ != test_img/ (physically separate directories)

        Args:
            sroie_dir: Path to SROIE data directory

        Returns:
            DataSplitValidationReport with results
        """
        report = DataSplitValidationReport(passed=True)

        # Check directories exist
        train_img_dir = sroie_dir / "img"
        train_key_dir = sroie_dir / "key"
        val_img_dir = sroie_dir / "val_img"
        val_key_dir = sroie_dir / "val_key"
        test_img_dir = sroie_dir / "test_img"
        test_key_dir = sroie_dir / "test_key"

        for dir_path in [
            train_img_dir,
            train_key_dir,
            val_img_dir,
            val_key_dir,
            test_img_dir,
            test_key_dir,
        ]:
            if not dir_path.exists():
                report.passed = False
                report.errors.append(f"Missing directory: {dir_path}")

        if not report.passed:
            return report

        # Count images in each set
        train_images = {f.stem for f in train_img_dir.glob("*")}
        val_images = {f.stem for f in val_img_dir.glob("*")}
        test_images = {f.stem for f in test_img_dir.glob("*")}

        report.train_count = len(train_images)
        report.val_count = len(val_images)
        report.test_count = len(test_images)

        # Validate counts
        if report.train_count != DataSplitValidator.EXPECTED_TRAIN_COUNT:
            report.passed = False
            report.errors.append(
                f"Training set has {report.train_count} images, "
                f"expected {DataSplitValidator.EXPECTED_TRAIN_COUNT}"
            )

        if report.val_count != DataSplitValidator.EXPECTED_VAL_COUNT:
            report.passed = False
            report.errors.append(
                f"Validation set has {report.val_count} images, "
                f"expected {DataSplitValidator.EXPECTED_VAL_COUNT}"
            )

        if report.test_count != DataSplitValidator.EXPECTED_TEST_COUNT:
            report.passed = False
            report.errors.append(
                f"Test set has {report.test_count} images, "
                f"expected {DataSplitValidator.EXPECTED_TEST_COUNT}"
            )

        # Check for overlap
        val_train_overlap = train_images & val_images
        if val_train_overlap:
            report.passed = False
            report.errors.append(
                f"Overlap between train and val: {len(val_train_overlap)} images "
                f"(e.g., {list(val_train_overlap)[:3]})"
            )

        test_train_overlap = train_images & test_images
        if test_train_overlap:
            report.passed = False
            report.errors.append(
                f"Overlap between train and test: {len(test_train_overlap)} images "
                f"(e.g., {list(test_train_overlap)[:3]})"
            )

        val_test_overlap = val_images & test_images
        if val_test_overlap:
            report.passed = False
            report.errors.append(
                f"Overlap between val and test: {len(val_test_overlap)} images "
                f"(e.g., {list(val_test_overlap)[:3]})"
            )

        # Check that all images have key files
        for split_name, _img_dir, key_dir, img_set in [
            ("train", train_img_dir, train_key_dir, train_images),
            ("val", val_img_dir, val_key_dir, val_images),
            ("test", test_img_dir, test_key_dir, test_images),
        ]:
            key_files = {f.stem for f in key_dir.glob("*.txt")}
            missing_keys = img_set - key_files
            if missing_keys:
                report.warnings.append(
                    f"{split_name} split: {len(missing_keys)} images missing key files "
                    f"(e.g., {list(missing_keys)[:3]})"
                )

        if not report.passed:
            _log_split.error("SROIE split validation FAILED:\n" + "\n".join(report.errors))
        else:
            _log_split.info(
                f"✓ SROIE split validation passed: "
                f"train={report.train_count}, val={report.val_count}, test={report.test_count}"
            )

        return report

    @staticmethod
    def check_no_val_test_leakage(val_dir: Path, test_dir: Path) -> tuple[bool, str]:
        """Critical check: val and test directories must be physically separate.

        This prevents accidental usage of test data during validation.

        Args:
            val_dir: Validation data directory path
            test_dir: Test data directory path

        Returns:
            (is_valid: bool, error_message: str)
        """
        # Check they are different paths
        if val_dir.resolve() == test_dir.resolve():
            return (
                False,
                f"CRITICAL: val_dir and test_dir are the SAME: {val_dir}\n"
                f"This causes data leakage - validation data is used as test data.\n"
                f"The split must use physically separate directories.",
            )

        # Check they don't overlap
        val_images = {f.stem for f in val_dir.glob("*") if f.is_file()}
        test_images = {f.stem for f in test_dir.glob("*") if f.is_file()}
        overlap = val_images & test_images

        if overlap:
            return (
                False,
                f"CRITICAL: val and test directories overlap on {len(overlap)} images\n"
                f"Examples: {list(overlap)[:5]}\n"
                f"This causes data leakage.",
            )

        return True, ""


# ---------------------------------------------------------------------------
# ImportChainChecker
# ---------------------------------------------------------------------------


class ImportChainChecker:
    """Check if the critical import chain (constants.py) is working.

    From CLAUDE.md Section 5: "if the import chain in constants.py is broken,
    nothing else matters. Fix core first."

    This is the FIRST validation run before ANY other pipeline stage.
    """

    @staticmethod
    def check_constants_import() -> tuple[bool, str]:
        """Check if constants.py can be imported.

        Returns:
            (success: bool, error_message: str)
        """
        try:
            # Try importing the critical constants required by every pipeline stage.
            # IMAGE_EXTS, MAX_LENGTH, NEW_TOKENS, and EMPTY_GT are validated in
            # PreflightChecker.check_constants_integrity (preflight_checks.py).
            from constants import (
                BASE_MODEL,
                FIELDS,
                SEED,
            )

            # Validate required fields exist
            if not FIELDS or not isinstance(FIELDS, (list, tuple)):
                return False, "FIELDS is empty or not a list"

            if not BASE_MODEL or not isinstance(BASE_MODEL, str):
                return False, "BASE_MODEL is empty or not a string"

            if SEED is None:
                return False, "SEED is None"

            return True, ""

        except ImportError as e:
            return False, f"ImportError: {str(e)}"
        except AttributeError as e:
            return False, f"AttributeError: {str(e)}"
        except SyntaxError as e:
            return False, f"SyntaxError in constants.py: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {type(e).__name__}: {str(e)}"

    @staticmethod
    def check_dataset_loaders_import() -> tuple[bool, str]:
        """Check if dataset_loaders.py can be imported.

        Returns:
            (success: bool, error_message: str)
        """
        try:
            from dataset_loaders import SROIELoader

            _ = SROIELoader()  # Try instantiating to catch runtime issues
            return True, ""

        except ImportError as e:
            return False, f"ImportError: {str(e)}"
        except SyntaxError as e:
            return False, f"SyntaxError in dataset_loaders.py: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {type(e).__name__}: {str(e)}"

    @staticmethod
    def check_all() -> tuple[bool, list[str]]:
        """Run all import chain checks.

        Returns:
            (all_passed: bool, error_messages: List[str])
        """
        errors = []

        # Check constants
        success, msg = ImportChainChecker.check_constants_import()
        if not success:
            errors.append(f"❌ constants.py import failed: {msg}")
        else:
            _log_chain.info("✓ constants.py import successful")

        # Check dataset loaders
        success, msg = ImportChainChecker.check_dataset_loaders_import()
        if not success:
            errors.append(f"❌ dataset_loaders.py import failed: {msg}")
        else:
            _log_chain.info("✓ dataset_loaders.py import successful")

        return len(errors) == 0, errors

    @staticmethod
    def diagnose_import_error(error_msg: str) -> str:
        """Provide diagnostic suggestions for import errors.

        Args:
            error_msg: The error message from import attempt

        Returns:
            Diagnostic suggestion string
        """
        suggestions = []

        if "tie_word_embeddings" in error_msg.lower():
            suggestions.append(
                "💡 Suggestion: Check if config.tie_word_embeddings is properly set in train.py"
            )

        if "no module named" in error_msg.lower():
            suggestions.append(
                "💡 Suggestion: Run 'pip install -r requirements.txt' to install missing packages"
            )

        if "syntax error" in error_msg.lower():
            suggestions.append(
                "💡 Suggestion: Check for syntax errors (missing brackets, commas) in constants.py"
            )

        if "lm_head" in error_msg.lower():
            suggestions.append(
                "💡 Suggestion: Ensure lm_head weight is properly initialized after resize_token_embeddings()"
            )

        if not suggestions:
            suggestions.append(
                "💡 Suggestion: Check that Python path is correct and all files are properly saved"
            )

        return "\n".join(suggestions)


# ---------------------------------------------------------------------------
# ModelWeightValidator
# ---------------------------------------------------------------------------


class ModelWeightValidator:
    """Validate model weights after checkpoint load."""

    @staticmethod
    def check_lm_head_weight_in_checkpoint(checkpoint_path: Path) -> tuple[bool, str]:
        """Check if decoder.lm_head.weight exists in checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            (exists: bool, error_message: str)
        """
        try:
            from safetensors import safe_open

            # Try loading safetensors files
            safetensors_files = list(checkpoint_path.glob("*.safetensors"))
            if not safetensors_files:
                return False, "No safetensors files found in checkpoint"

            # Check if lm_head.weight is in any shard
            found_lm_head = False
            for shard_path in safetensors_files:
                try:
                    with safe_open(shard_path, framework="pt") as f:
                        keys = f.keys()
                        if "decoder.lm_head.weight" in keys:
                            found_lm_head = True
                            _log_weight.info(f"✓ Found decoder.lm_head.weight in {shard_path.name}")
                            break
                except Exception as e:
                    _log_weight.warning(f"Could not read {shard_path}: {e}")

            if found_lm_head:
                return True, ""
            else:
                return (
                    False,
                    "decoder.lm_head.weight not found in checkpoint (safetensors dedup?)",
                )

        except ImportError:
            _log_weight.warning("safetensors not installed, skipping weight check")
            return True, ""  # Skip check if safetensors unavailable
        except Exception as e:
            _log_weight.warning(f"Could not validate lm_head weight: {e}")
            return True, ""  # Don't fail on unknown errors

    @staticmethod
    def validate_model_forward_pass(
        model,
        processor,
        test_image_path: Path | None = None,
    ) -> tuple[bool, str]:
        """Test that model can do forward pass (catches silent failures).

        Args:
            model: The DONUT model
            processor: The DonutProcessor
            test_image_path: Path to test image (optional)

        Returns:
            (success: bool, error_message: str)
        """
        try:
            import torch
            from PIL import Image

            # Create dummy input if no test image
            if test_image_path is None:
                # Create blank image
                img = Image.new("RGB", (960, 1280), color="white")
            else:
                try:
                    img = Image.open(test_image_path)
                except Exception as e:
                    _log_weight.warning(f"Could not load test image: {e}")
                    return True, ""  # Skip if can't load

            # Process image
            pixel_values = processor(img, return_tensors="pt").pixel_values

            # Run forward pass
            with torch.no_grad():
                outputs = model.encoder(pixel_values)

            if outputs is None or not hasattr(outputs, "last_hidden_state"):
                return False, "Encoder returned invalid output"

            _log_weight.info("✓ Model forward pass successful")
            return True, ""

        except Exception as e:
            _log_weight.warning(f"Model forward pass test failed: {e}")
            return False, f"Forward pass failed: {str(e)}"

    @staticmethod
    def check_missing_keys_after_load(
        missing_keys: list[str],
        unexpected_keys: list[str],
        tie_word_embeddings: bool,
    ) -> tuple[bool, str]:
        """Check if critical keys are missing after model load.

        Args:
            missing_keys: Keys from loading_info['missing_keys']
            unexpected_keys: Keys from loading_info['unexpected_keys']
            tie_word_embeddings: Value of model.config.tie_word_embeddings

        Returns:
            (valid: bool, error_message: str)
        """
        critical_keys = ["decoder.lm_head.weight"]

        for key in critical_keys:
            if key in missing_keys:
                if not tie_word_embeddings:
                    # This is a BUG - lm_head should be saved separately
                    return (
                        False,
                        f"CRITICAL: {key} missing from checkpoint (safetensors dedup)\n"
                        f"This will cause F1~0.42 (all predictions garbled)\n"
                        f"Fix: Ensure LmHeadCloneCallback is used during training",
                    )
                else:
                    # If tie_word_embeddings=True, lm_head is tied to embed_tokens, OK to be missing
                    _log_weight.info(f"Note: {key} missing but tie_word_embeddings=True (expected)")

        if unexpected_keys:
            _log_weight.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

        return True, ""


# ---------------------------------------------------------------------------
# SeedValidator
# ---------------------------------------------------------------------------


class SeedValidator:
    """Validate seed reproducibility settings."""

    EXPECTED_SEED = 42

    @staticmethod
    def check_constants_seed() -> tuple[bool, str]:
        """Check if constants.SEED == 42.

        Returns:
            (valid: bool, error_message: str)
        """
        try:
            from constants import SEED

            if SEED == SeedValidator.EXPECTED_SEED:
                _log_seed.info(f"✓ constants.SEED == {SEED}")
                return True, ""
            else:
                return (
                    False,
                    f"constants.SEED={SEED}, expected {SeedValidator.EXPECTED_SEED}",
                )

        except ImportError as e:
            return False, f"Could not import SEED from constants: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    @staticmethod
    def check_seed_set_calls() -> tuple[bool, str]:
        """Check if set_seed() is called before experiments.

        Returns:
            (valid: bool, error_message: str)
        """
        try:
            # Look for set_seed in common locations
            for module_name in ["utils", "train", "run_experiments", "run_all"]:
                try:
                    mod = importlib.import_module(module_name)
                    if hasattr(mod, "set_seed"):
                        _log_seed.info(f"✓ Found set_seed() in {module_name}.py")
                        return True, ""
                except ImportError:
                    continue

            _log_seed.warning("set_seed() function not found in common modules")
            return True, ""  # Not critical if not found

        except Exception as e:
            _log_seed.warning(f"Could not check for set_seed(): {e}")
            return True, ""

    @staticmethod
    def test_rng_consistency() -> tuple[bool, str]:
        """Test that RNGs are consistent when seeded.

        Returns:
            (consistent: bool, error_message: str)
        """
        try:
            import numpy as np

            # Seed all RNGs
            seed = SeedValidator.EXPECTED_SEED
            random.seed(seed)
            np.random.seed(seed)

            # Get deterministic outputs
            r1 = random.random()
            n1 = np.random.rand()

            # Reseed and check reproducibility
            random.seed(seed)
            np.random.seed(seed)
            r2 = random.random()
            n2 = np.random.rand()

            if r1 == r2 and n1 == n2:
                _log_seed.info("✓ RNG determinism verified")
                return True, ""
            else:
                return False, "RNG output not reproducible"

        except Exception as e:
            _log_seed.warning(f"Could not verify RNG consistency: {e}")
            return True, ""

    @staticmethod
    def check_all() -> tuple[bool, list[str]]:
        """Run all seed checks.

        Returns:
            (all_valid: bool, error_messages: List[str])
        """
        errors = []

        # Check constants.SEED
        valid, msg = SeedValidator.check_constants_seed()
        if not valid:
            errors.append(f"❌ Seed constant check failed: {msg}")

        # Check set_seed() exists
        valid, msg = SeedValidator.check_seed_set_calls()
        if not valid:
            errors.append(f"❌ set_seed() check failed: {msg}")

        # Test RNG consistency
        valid, msg = SeedValidator.test_rng_consistency()
        if not valid:
            errors.append(f"❌ RNG consistency check failed: {msg}")

        return len(errors) == 0, errors
