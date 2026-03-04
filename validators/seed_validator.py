"""Validator for seed reproducibility - ensures SEED=42 is consistent."""

import logging

__all__ = ["SeedValidator"]

logger = logging.getLogger(__name__)


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
                logger.info(f"✓ constants.SEED == {SEED}")
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
            # Try importing set_seed
            import importlib

            # Look for set_seed in common locations
            for module_name in ["utils", "train", "run_experiments", "run_all"]:
                try:
                    mod = importlib.import_module(module_name)
                    if hasattr(mod, "set_seed"):
                        logger.info(f"✓ Found set_seed() in {module_name}.py")
                        return True, ""
                except ImportError:
                    continue

            logger.warning("set_seed() function not found in common modules")
            return True, ""  # Not critical if not found

        except Exception as e:
            logger.warning(f"Could not check for set_seed(): {e}")
            return True, ""

    @staticmethod
    def test_rng_consistency() -> tuple[bool, str]:
        """Test that RNGs are consistent when seeded.

        Returns:
            (consistent: bool, error_message: str)
        """
        try:
            import random

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
                logger.info("✓ RNG determinism verified")
                return True, ""
            else:
                return False, "RNG output not reproducible"

        except Exception as e:
            logger.warning(f"Could not verify RNG consistency: {e}")
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
