"""Validator for checking the import chain - CRITICAL first check."""

import logging

__all__ = ["ImportChainChecker"]

logger = logging.getLogger(__name__)


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
            logger.info("✓ constants.py import successful")

        # Check dataset loaders
        success, msg = ImportChainChecker.check_dataset_loaders_import()
        if not success:
            errors.append(f"❌ dataset_loaders.py import failed: {msg}")
        else:
            logger.info("✓ dataset_loaders.py import successful")

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
