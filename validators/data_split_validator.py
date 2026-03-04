"""Validator for SROIE data split integrity - prevents data leakage.

From CLAUDE.md Section 16 - BUG A:
"Data leakage in eval: Separate val_img/ and test_img/ directories in stage_install()
val split missing → no early stopping guard"

This validator ensures the critical 80/10/10 split is maintained with physical
directory separation to prevent accidental use of test data during training.
"""

import logging
from pathlib import Path

from pipeline_types import DataSplitValidationReport

logger = logging.getLogger(__name__)


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

        for dir_path in [train_img_dir, train_key_dir, val_img_dir, val_key_dir, test_img_dir, test_key_dir]:
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
        for split_name, img_dir, key_dir, img_set in [
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
            logger.error("SROIE split validation FAILED:\n" + "\n".join(report.errors))
        else:
            logger.info(
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
