"""
pipeline_types/exceptions.py — Pipeline-level exception classes.

Defines exceptions that can be raised at various stages of the DONUT
Receipt KIE pipeline.  Keeping them in pipeline_types/ makes them
importable project-wide without depending on any heavy third-party libs.
"""
from __future__ import annotations


class CheckpointCorruptionError(Exception):
    """Raised when a model checkpoint fails integrity validation.

    Possible causes:
    - ``lm_head.weight`` is missing from the checkpoint (safetensors
      deduplication dropped it because it was tied to ``embed_tokens``).
    - ``model.config.encoder.image_size`` does not match the image size
      used during training (stored in the experiment result JSON).
    - The token vocabulary size in the checkpoint's embedding matrix does
      not match the expected size after ``add_special_tokens()``.

    See: CLAUDE.md § 5 (Pattern 6) and validators/checkpoint_resume_validator.py.
    """


class PipelineConfigError(Exception):
    """Raised when experiment or pipeline configuration is invalid."""


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded or validated."""
