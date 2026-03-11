"""Root conftest.py — ensures the project root is on sys.path for all pytest runs.

This file is auto-loaded by pytest before any test file is collected.
It replaces the duplicated ``sys.path.insert(0, ...)`` boilerplate that used
to appear at the top of every test file.

Note: ``pyproject.toml`` already sets ``pythonpath = ["."]`` for pytest, so
this file is belt-and-suspenders insurance that also works when conftest.py is
loaded explicitly or from a non-standard working directory.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
