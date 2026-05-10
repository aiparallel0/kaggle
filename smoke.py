"""Offline-aware smoke-test wrapper around diagnostics.smoke_test.

The issues that fired #150-#154 came from `python diagnostics.py --smoke-test`
being run repeatedly before the in-process + cross-process dedup landed in
`diagnostics.py`. The dedup is now in place, but the underlying smoke test
still loads ~1GB of weights three separate times and aborts hard if the host
lacks internet or disk space. This wrapper:

* runs the offline-safe checks first (imports, constants, token-ID logic),
* short-circuits on `--offline` so users without HF access can verify the
  Python paths,
* forwards to the full `diagnostics.smoke_test` only when the cheap checks pass
  AND we're not in offline mode.

Usage:
    python3 smoke.py            # full smoke test (downloads model)
    python3 smoke.py --offline  # skip model-loading checks
    python3 smoke.py --quick    # only checks 1-3 (no forward pass)
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback


def _check_imports() -> None:
    from constants import BASE_MODEL, FIELDS, NEW_TOKENS, SEED  # noqa: F401
    from data_pipeline import SROIELoader  # noqa: F401


def _check_token_logic_offline() -> None:
    """Sanity-check that constants are well-formed without loading weights."""
    from constants import NEW_TOKENS

    assert isinstance(NEW_TOKENS, list) and NEW_TOKENS, "NEW_TOKENS must be a non-empty list"
    assert "<s_sroie>" in NEW_TOKENS, "<s_sroie> missing from NEW_TOKENS (GP-3 prereq)"
    assert all(isinstance(t, str) and t.startswith("<") and t.endswith(">") for t in NEW_TOKENS), (
        "every NEW_TOKEN must be a <tag>-form string"
    )


def _run_full() -> bool:
    from diagnostics import smoke_test
    return smoke_test(verbose=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="Skip checks that need to download weights")
    parser.add_argument("--quick", action="store_true", help="Imports + token logic only, no forward pass")
    args = parser.parse_args(argv)

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("=" * 60)
    print(f"SMOKE TEST  (offline={args.offline}, quick={args.quick})")
    print("=" * 60)

    try:
        print("[1/2] Imports + constants...")
        _check_imports()
        _check_token_logic_offline()
        print("      ok")
    except Exception as exc:  # noqa: BLE001
        print(f"      FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    if args.offline or args.quick:
        print("[2/2] Skipped (offline or quick mode)")
        print("\nNote: this only verifies the offline-safe paths. Run without --offline")
        print("      to exercise the full diagnostics.smoke_test.")
        return 0

    print("[2/2] Full diagnostics.smoke_test...")
    ok = _run_full()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
