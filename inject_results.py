# Re-export shim — logic moved to reporting.py
# Kept for backward compatibility with DO NOT TOUCH late imports in:
#   run_all.py (lines 1996, 2676)
from reporting import *  # noqa: F401, F403
