# Re-export shim — logic moved to cloud_orchestration.py
# Kept for backward compatibility: CritiqueReport.print_loud() imports this
# module at runtime via importlib.import_module("pipeline_critic").
from cloud_orchestration import _print_report  # noqa: F401
