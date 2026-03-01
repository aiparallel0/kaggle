# MIT License
#
# Copyright (c) 2024
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
evaluate.py — Backward-compatibility shim + validation utility.

This module was originally renamed to donut_evaluator.py to avoid shadowing the
HuggingFace ``evaluate`` library. This file maintains backward compatibility by
re-exporting all public names from donut_evaluator.py, and adds validation
utilities for pipeline health checks.

Use this module for:
  1. Import compatibility: `from evaluate import DonutEvaluator`
  2. Pipeline validation: `python evaluate.py --validate`
  3. Model health checks: `python evaluate.py --check-model path/to/model`
"""

import sys
from pathlib import Path

from donut_evaluator import *  # noqa: F401, F403
from donut_evaluator import (  # noqa: F401  (explicit for static analysers)
    _DIAGNOSTIC_LOG_COUNT,
    DEVICE,
    DonutEvaluator,
    EvaluationResult,
    _retie_decoder_head,
    _unwrap_prediction,
    compute_metrics,
    load_model_with_tied_weights,
    normalized_edit_distance,
    print_results,
    remap_cord_to_sroie,
    run_inference,
)


def validate_pipeline() -> bool:
    """Validate that the evaluation pipeline can load and initialize models.

    Returns True if all checks pass, False otherwise.
    """
    print("\n🔍 Validating evaluation pipeline...\n")

    checks = {
        "constants": False,
        "donut_evaluator": False,
        "device": False,
        "device_type": False,
    }

    # Check constants import
    try:
        from constants import FIELDS, BASE_MODEL, MAX_LENGTH
        print(f"  ✓ Constants loaded: {len(FIELDS)} fields, max_length={MAX_LENGTH}")
        checks["constants"] = True
    except Exception as e:
        print(f"  ✗ Failed to load constants: {e}")

    # Check donut_evaluator import
    try:
        assert DonutEvaluator is not None
        print(f"  ✓ DonutEvaluator class available")
        checks["donut_evaluator"] = True
    except Exception as e:
        print(f"  ✗ Failed to load DonutEvaluator: {e}")

    # Check device detection
    try:
        import torch
        print(f"  ✓ PyTorch loaded: {torch.__version__}")
        checks["device"] = True
    except Exception as e:
        print(f"  ✗ Failed to load PyTorch: {e}")

    # Check DEVICE constant
    try:
        print(f"  ✓ Detected device: {DEVICE}")
        checks["device_type"] = True
    except Exception as e:
        print(f"  ✗ Failed to detect device: {e}")

    print(f"\n{'='*50}")
    if all(checks.values()):
        print("✅ Pipeline validation PASSED")
        return True
    else:
        print("❌ Pipeline validation FAILED:")
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"   {status} {check}")
        return False


def check_model_integrity(model_path: str) -> bool:
    """Check if a saved model can be loaded and has required weight tensors.

    Args:
        model_path: Path to the model directory or checkpoint

    Returns:
        True if model is valid, False otherwise.
    """
    print(f"\n🔧 Checking model integrity: {model_path}\n")

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"  ✗ Model path does not exist: {model_path}")
        return False

    try:
        model = load_model_with_tied_weights(str(model_path), device=DEVICE)
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Model dtype: {model.dtype}")
        print(f"  ✓ Model device: {next(model.parameters()).device}")

        # Check for critical weights
        has_encoder = hasattr(model, "encoder") and model.encoder is not None
        has_decoder = hasattr(model, "decoder") and model.decoder is not None
        has_lm_head = hasattr(model.decoder, "lm_head") if has_decoder else False

        print(f"  ✓ Has encoder: {has_encoder}")
        print(f"  ✓ Has decoder: {has_decoder}")
        print(f"  ✓ Has lm_head: {has_lm_head}")

        if all([has_encoder, has_decoder, has_lm_head]):
            print(f"\n✅ Model integrity check PASSED")
            return True
        else:
            print(f"\n❌ Model missing critical components")
            return False

    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluation utilities and validation checks"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the evaluation pipeline setup"
    )
    parser.add_argument(
        "--check-model",
        type=str,
        metavar="PATH",
        help="Check integrity of a saved model"
    )

    args = parser.parse_args()

    if args.validate:
        success = validate_pipeline()
        sys.exit(0 if success else 1)
    elif args.check_model:
        success = check_model_integrity(args.check_model)
        sys.exit(0 if success else 1)
    else:
        print(__doc__)
        parser.print_help()
