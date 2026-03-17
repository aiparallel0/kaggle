# Re-export shim — logic moved to experiment_config.py
# Kept for backward compatibility with DO NOT TOUCH imports in:
#   train_trocr_yolo.py, run_experiments.py
from experiment_config import *  # noqa: F401, F403
from experiment_config import (
    __all__,  # noqa: F401 (re-export __all__ so control_suite.__all__ works)
)
