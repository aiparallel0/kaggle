"""tests/conftest.py — sys.path bootstrap for the test suite.

Loaded by pytest before any test in the tests/ directory is collected.
Ensures the project root is on sys.path so imports like ``from constants import …``
work without each test file needing its own ``sys.path.insert`` call.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
