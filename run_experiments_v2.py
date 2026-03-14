# =============================================================================
# run_experiments_v2.py  — DEPRECATION SHIM
# =============================================================================
"""
run_experiments_v2.py — Compatibility redirect.

All v2 functionality (resolution-sync fix, finetune_height/finetune_width
fields, _apply_resolution_sync()) has been merged into run_experiments.py.

Import everything from run_experiments going forward.  This file is kept
solely so existing code that imports from run_experiments_v2 does not break.
"""

from run_experiments import (  # noqa: F401  (re-exports for backward compat)
    EXPERIMENTS,
    RESULTS_DIR,
    TRAIN_CONFIG,
    ExperimentConfig,
    _apply_resolution_sync,
    _config_to_dict,
    run_custom_experiment,
    run_experiment,
    run_experiment_from_config,
    save_summary,
)

import warnings as _warnings

_warnings.warn(
    "run_experiments_v2 is deprecated. Import from run_experiments instead.",
    DeprecationWarning,
    stacklevel=2,
)
