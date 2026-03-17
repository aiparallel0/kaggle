# Re-export shim — logic moved to resource_manager.py
# Kept for backward compatibility with DO NOT TOUCH imports in:
#   train.py, run_experiments.py
from resource_manager import *  # noqa: F401, F403
