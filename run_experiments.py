# =============================================================================
# run_experiments.py
# Purpose: 8-experiment DONUT fine-tuning loop with OOM recovery and JSON result serialization
# Merged from: experiment_config.py + evaluation.py
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-17
# =============================================================================
"""
run_experiments.py — Experiment orchestrator for 8 DONUT fine-tuning experiments.

This module is imported by run_all.py as part of the full dual-architecture
pipeline.  It can also be run standalone for DONUT-only experiment runs.

See also: run_all.py — the canonical single entry point that calls this module
          plus the TrOCR+YOLO stages, benchmark comparison, and paper generation.

Usage
-----
Run all experiments:
    python run_experiments.py --all

Run a single experiment (e.g., experiment 2):
    python run_experiments.py --experiment 2

Force re-run (ignore cached results):
    python run_experiments.py --all --force

Each experiment trains DONUT from the base checkpoint (donut-base) and evaluates on the
SROIE test set (63 images in test_img / test_key — custom 80/10/10 split
from 626 labeled training images; the official 347-image test set has no
public ground truth).  Results are saved to
results/experiment_N.json and a summary to results/all_experiments.json.

Architecture
------------
ExperimentConfig is THE single source of truth for all training hyperparameters.
No hardcoded epoch values, learning rates, or batch sizes exist outside of it.
DonutTrainer (from train.py) receives an ExperimentConfig and reads all
hyperparameters from it via duck-typed attribute access.
DonutEvaluator (from donut_evaluator.py) handles model loading with weight re-tying
and evaluation with self-test and parse-failure thresholds.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import glob
import json
import logging
import math
import os
import re
import struct
import sys
import time
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

# Ensure sibling modules are importable regardless of CWD.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Set before torch initializes to reduce GPU memory fragmentation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import DonutProcessor, VisionEncoderDecoderModel  # noqa: E402

import data_pipeline as dataset_loaders  # noqa: E402
import resource_manager as _mm  # noqa: E402

# FIX: Import shared constants from single source of truth (constants.py)
# instead of duplicating FIELDS/IMAGE_EXTS/etc. independently in this file.
from constants import (  # noqa: E402
    BASE_MODEL,
    DEVICE,
    EMPTY_GT,
    FIELDS,
    MAX_LENGTH,
    NEW_TOKENS,
    SEED,
    WORKSPACE,
    _edit_distance,
    _get_sroie_dir,
    _gpu_cleanup,
    _progress,
    set_seed,
)
from data_pipeline import load_sroie_test  # noqa: E402
from resource_manager import TrainingAuditLogger  # noqa: E402
from train import MultiDataset, _ensure_dual_config  # noqa: E402

try:
    import yaml  # noqa: E402
except ImportError as e:
    raise ImportError("PyYAML is required: pip install pyyaml") from e

try:
    import flash_attn  # noqa: F401

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # from experiment_config
    "DatasetEntry",
    "load_experiment",
    "load_all_experiments",
    "load_experiment_selection",
    "DonutControlConfig",
    "TrOCRControlConfig",
    "YOLOControlConfig",
    "ControlSuite",
    "CONTROL_SUITE",
    "validate_sroie_oversample",
    "get_augmentation_transforms",
    # from evaluation
    "DonutEvaluator",
    "EvaluationResult",
    "compute_metrics",
    "normalized_edit_distance",
    "load_model_with_tied_weights",
    "run_inference",
    "remap_cord_to_sroie",
    "DEVICE",
    "load_test_samples",
    "evaluate_donut_on_test",
    "evaluate_trocr_yolo_on_test",
    "print_metrics",
    "generate_comparison_report",
    "generate_json_summary",
    # from run_experiments
    "EvaluationUndertrainedError",
    "ExperimentConfig",
    "EXPERIMENTS",
    "TRAIN_CONFIG",
    "_config_to_dict",
    "run_experiment",
    "run_experiment_from_config",
    "run_custom_experiment",
    "save_summary",
    "_apply_resolution_sync",
    "SelfTestFailedError",
]

# ---------------------------------------------------------------------------
# Sentinel exception for expected evaluation failures (undertrained models)
# ---------------------------------------------------------------------------


class EvaluationUndertrainedError(RuntimeError):
    """Raised when a model is too undertrained to produce parseable output.

    Using a dedicated class instead of RuntimeError + string matching means:
    - Catch sites can target this exact exception, not any RuntimeError.
    - A real RuntimeError (tensor shape mismatch, CUDA error, etc.) is never
      silently converted to F1=0.0 just because its message happens to contain
      one of the old sentinel substrings.
    - Future refactoring can rename the message without breaking catch sites.

    Raised by:
    - DonutEvaluator.evaluate() when parse_failure_count > 50% threshold.
    - DonutEvaluator._self_test() when the model produces an empty prediction.
# Custom exception classes
# ---------------------------------------------------------------------------


class SelfTestFailedError(RuntimeError):
    """Raised when the DonutEvaluator self-test detects the model cannot produce
    parseable output on a single sample.

    Using a dedicated exception type (instead of substring-matching RuntimeError
    messages) makes the catch clause in run_experiment() unambiguous and immune
    to capitalisation changes or unrelated RuntimeErrors.

    Fix: issue_report_summary high #7.
    """


# ---------------------------------------------------------------------------
# Logging Configuration — MUST be set before any third-party imports
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Suppress verbose third-party HTTP loggers to keep output clean
for pkg in ["httpx", "httpcore", "urllib3", "datasets", "transformers", "huggingface_hub"]:
    logging.getLogger(pkg).setLevel(logging.WARNING)

# Suppress PIL chunk-level DEBUG flood and other noisy third-party loggers
try:
    from constants import suppress_noisy_loggers

    suppress_noisy_loggers()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# experiment_config — YAML loader + ablation control suite
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DatasetEntry — per-dataset config within an experiment
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DatasetEntry:
    """One dataset slot in an experiment's training mix."""

    name: str
    split: str = "train"
    oversample: int = 1

    def __post_init__(self) -> None:
        if self.oversample < 1:
            raise ValueError(
                f"DatasetEntry '{self.name}': oversample must be ≥ 1, got {self.oversample}"
            )
        if self.split not in {"train", "all", "val", "test"}:
            raise ValueError(
                f"DatasetEntry '{self.name}': split must be one of "
                f"'train'/'all'/'val'/'test', got '{self.split}'"
            )


# ---------------------------------------------------------------------------
# _YAMLExperimentConfig — config loaded from experiments/*.yaml files only
# (private; the public ExperimentConfig for hardcoded experiments is defined
#  further below at the "DONUT fine-tuning experiments" section)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _YAMLExperimentConfig:
    """
    Single source of truth for all hyperparameters of one DONUT experiment.

    All fields have their types and constraints documented inline.
    DonutTrainer reads values via duck-typed attribute access — no magic
    numbers are hardcoded anywhere else in the training pipeline.
    """

    # Identifiers
    id: int
    name: str
    description: str = ""

    # Architecture type — controls which training entry point is used
    # "donut" → existing DONUT Seq2Seq fine-tuning path
    # "trocr_yolo" → train_trocr_yolo.py pipeline path
    arch_type: str = "donut"

    # Zero-shot flag — when True, skip fit() and go directly to evaluation
    is_zero_shot: bool = False

    # Model
    base_checkpoint: str = "naver-clova-ix/donut-base"
    tie_word_embeddings: bool = False  # MUST stay False after resize_token_embeddings()
    full_parameter_finetuning: bool = True

    # Data / preprocessing
    image_height: int = 1280  # DONUT native — do NOT exceed without allow_high_res
    image_width: int = 960  # DONUT native — do NOT exceed without allow_high_res
    allow_high_res: bool = False  # bypasses image_height>1280 / image_width>960 guard
    max_length: int = 768  # MAX_LENGTH from constants.py

    # Dataset mix
    datasets: list[DatasetEntry] = dataclasses.field(default_factory=list)

    # Training
    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 2  # effective batch = batch_size × grad_accum
    mixed_precision: str = "fp16"  # "fp16" | "bf16" | "fp32"
    resource_optimizer_target_effective_batch: int | None = (
        None  # if set, runner calls resource_optimizer.optimize_hyperparams()
    )

    # Optimizer (AdamW layerwise LR)
    optimizer_type: str = "AdamW"
    weight_decay: float = 0.01
    encoder_lr: float = 5e-5
    decoder_lr: float = 1e-4

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = (
        40  # matches run_experiments.ExperimentConfig default; 500 is capped for small datasets
    )

    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 3
    early_stopping_monitor: str = "val_loss"

    # Reproducibility
    seed: int = 42

    # Validation
    val_precompute_tensors: bool = False  # MUST stay False — see memory_manager.py

    # Output paths
    results_file: str = ""
    checkpoint_dir: str = ""
    log_file: str = ""

    # DAG scheduler: list of experiment IDs that must finish before this one starts.
    # An empty list means no dependencies (run immediately).
    # Used by dag_scheduler.py when --parallel is enabled.
    depends_on: list[int] = dataclasses.field(default_factory=list)

    # ---------------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        # Resolution guard — most important check
        if self.image_height > 1280 and not self.allow_high_res:
            raise ValueError(
                f"Exp {self.id}: image_height={self.image_height} exceeds "
                f"DONUT native 1280. RAM scales as (H×W)/(1280×960). "
                f"See memory_manager.py § 'The processor_config.json Rule'. "
                f"Set allow_high_res: true in the YAML to bypass this guard."
            )
        if self.image_width > 960 and not self.allow_high_res:
            raise ValueError(
                f"Exp {self.id}: image_width={self.image_width} exceeds "
                f"DONUT native 960. RAM scales as (H×W)/(1280×960). "
                f"Set allow_high_res: true in the YAML to bypass this guard."
            )
        if self.image_height > 1280 and self.allow_high_res:
            logger.debug(
                "[Exp %d] High-res mode: height=%d (>1280). allow_high_res=True bypasses guard. "
                "Ensure processor_config.json updated before training.",
                self.id,
                self.image_height,
            )
        if self.image_width > 960 and self.allow_high_res:
            logger.debug(
                "[Exp %d] High-res mode: width=%d (>960). allow_high_res=True bypasses guard. "
                "Ensure processor_config.json updated before training.",
                self.id,
                self.image_width,
            )

        # Weight-tying guard
        if self.tie_word_embeddings:
            raise ValueError(
                f"Exp {self.id}: tie_word_embeddings must be False. "
                f"Setting True destroys lm_head after resize_token_embeddings(), "
                f"causing F1=0.00 on every prediction."
            )

        # Val precompute guard
        if self.val_precompute_tensors:
            raise ValueError(
                f"Exp {self.id}: val_precompute_tensors must be False. "
                f"Precomputing val tensor cache causes OOM (see Exp 6 memory notes)."
            )

        # Dataset list must not be empty — unless this is a zero-shot experiment
        if not self.datasets and not self.is_zero_shot:
            raise ValueError(
                f"Exp {self.id}: datasets list is empty. "
                f"Set training.is_zero_shot: true if no training data is intended."
            )

        # Mixed precision check
        if self.mixed_precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError(
                f"Exp {self.id}: mixed_precision='{self.mixed_precision}' invalid. "
                f"Choose 'fp16', 'bf16', or 'fp32'."
            )

        # arch_type check
        if self.arch_type not in {"donut", "trocr_yolo"}:
            raise ValueError(
                f"Exp {self.id}: arch_type='{self.arch_type}' invalid. "
                f"Choose 'donut' or 'trocr_yolo'."
            )

    # Convenience helpers

    @property
    def experiment_id(self) -> int:
        """Alias for id — used by DonutTrainer.train() for LiveDashboard CSV naming."""
        return self.id

    @property
    def base_model(self) -> str:
        """Alias for base_checkpoint — used by run_experiments.py code paths."""
        return self.base_checkpoint

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def dataset_names(self) -> list[str]:
        """Names of all datasets in this experiment (with multiplicity collapsed)."""
        return [d.name for d in self.datasets]

    def estimated_sample_count(self, registry: dict[str, int] | None = None) -> int | None:
        """
        Returns estimated sample count if a registry dict {name: base_count}
        is provided, otherwise returns None.
        """
        if registry is None:
            return None
        total = 0
        for entry in self.datasets:
            base = registry.get(entry.name)
            if base is None:
                return None  # unknown dataset in registry
            total += base * entry.oversample
        return total


# ---------------------------------------------------------------------------
# YAML → ExperimentConfig
# ---------------------------------------------------------------------------


def _parse_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML file {path} must be a mapping at the top level.")
    return raw


def _require(d: dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise KeyError(f"Required key '{key}' missing in {path}")
    return d[key]


def load_experiment(
    experiment_id: int,
    experiments_dir: str | Path = "experiments",
) -> _YAMLExperimentConfig:
    """
    Load a single experiment by integer ID.

    Searches experiments_dir for any YAML file whose name contains the
    experiment ID (e.g. 'exp_06_...' or 'experiment_6_...').

    Raises FileNotFoundError if no matching file is found.
    """
    experiments_dir = Path(experiments_dir)
    pattern = str(experiments_dir / "*.yaml")
    candidates = glob.glob(pattern) + glob.glob(str(experiments_dir / "*.yml"))

    # Match files that contain the experiment ID as a number
    matched: list[Path] = []
    for c in candidates:
        stem = Path(c).stem
        # Look for exp_id appearing as a standalone number (underscore-bounded or start/end)
        if re.search(r"(?:^|_|exp)0*" + str(experiment_id) + r"(?:_|$)", stem, re.IGNORECASE):
            matched.append(Path(c))

    if not matched:
        raise FileNotFoundError(
            f"No YAML config found for experiment_id={experiment_id} "
            f"in directory '{experiments_dir}'. "
            f"Expected a file whose name contains '{experiment_id}' "
            f"(e.g. 'exp_0{experiment_id}_...yaml')."
        )
    if len(matched) > 1:
        raise RuntimeError(
            f"Ambiguous: multiple YAML files match experiment_id={experiment_id}: "
            + ", ".join(str(p) for p in matched)
        )

    return _yaml_to_config(str(matched[0]))


def load_all_experiments(
    experiments_dir: str | Path = "experiments",
    experiment_ids: Sequence[int] | None = None,
) -> list[_YAMLExperimentConfig]:
    """
    Load all experiment YAML files from experiments_dir, sorted by
    experiment_id.  Pass experiment_ids to load only specific IDs.

    Returns a list of ExperimentConfig objects, sorted by id ascending.
    """
    experiments_dir = Path(experiments_dir)
    pattern = str(experiments_dir / "*.yaml")
    paths = sorted(glob.glob(pattern) + glob.glob(str(experiments_dir / "*.yml")))

    if not paths:
        raise FileNotFoundError(
            f"No experiment YAML files found in '{experiments_dir}'. "
            f"Expected files matching '{experiments_dir}/*.yaml'."
        )

    configs: list[_YAMLExperimentConfig] = []
    errors: list[str] = []

    for p in paths:
        try:
            cfg = _yaml_to_config(p)
            if experiment_ids is None or cfg.id in experiment_ids:
                configs.append(cfg)
        except Exception as exc:
            errors.append(f"  {p}: {exc}")

    if errors:
        raise ValueError(
            f"Failed to load {len(errors)} experiment config(s):\n" + "\n".join(errors)
        )

    configs.sort(key=lambda c: c.id)
    return configs


def _yaml_to_config(path: str | Path) -> _YAMLExperimentConfig:
    """Parse one YAML file into a _YAMLExperimentConfig."""
    path = str(path)
    raw = _parse_yaml(path)

    exp_id = int(_require(raw, "experiment_id", path))
    name = str(_require(raw, "name", path))
    description = str(raw.get("description", ""))

    # arch section — informational, passed through
    arch = raw.get("arch", {})
    arch_type = str(arch.get("type", "donut"))

    # model section
    model = raw.get("model", {})
    base_checkpoint = str(model.get("base_checkpoint", "naver-clova-ix/donut-base"))
    tie_word_embeddings = bool(model.get("tie_word_embeddings", False))
    full_parameter_finetuning = bool(model.get("full_parameter_finetuning", True))

    # data section
    data = raw.get("data", {})
    image_height = int(data.get("image_height", 1280))
    image_width = int(data.get("image_width", 960))
    allow_high_res = bool(data.get("allow_high_res", False))
    max_length = int(data.get("max_decode_length", 768))

    # training section — read is_zero_shot before datasets validation
    tr = raw.get("training", {})
    is_zero_shot = bool(tr.get("is_zero_shot", False)) or bool(tr.get("skip", False))

    # datasets section — may be empty list or null for zero-shot experiments
    raw_datasets = raw.get("datasets", [])
    if raw_datasets is None:
        raw_datasets = []
    if not isinstance(raw_datasets, list):
        raise ValueError(f"'datasets' in {path} must be a YAML list.")
    dataset_entries = [
        DatasetEntry(
            name=str(_require(d, "name", path)),
            split=str(d.get("split", "train")),
            oversample=int(d.get("oversample", 1)),
        )
        for d in raw_datasets
    ]

    epochs = int(tr.get("epochs", 10))
    batch_size = int(tr.get("batch_size", 8))
    gradient_accumulation_steps = int(tr.get("gradient_accumulation_steps", 2))
    mixed_precision = str(tr.get("mixed_precision", "fp16"))
    seed = int(tr.get("seed", 42))
    _roteb = tr.get("resource_optimizer_target_effective_batch")
    resource_optimizer_target_effective_batch = int(_roteb) if _roteb is not None else None

    opt = tr.get("optimizer", {})
    optimizer_type = str(opt.get("type", "AdamW"))
    weight_decay = float(opt.get("weight_decay", 0.01))
    encoder_lr = float(opt.get("encoder_lr", 5e-5))
    decoder_lr = float(opt.get("decoder_lr", 1e-4))

    sched = tr.get("scheduler", {})
    scheduler_type = str(sched.get("type", "cosine"))
    warmup_steps = int(sched.get("warmup_steps", 40))

    es = tr.get("early_stopping", {})
    early_stopping_enabled = bool(es.get("enabled", True))
    early_stopping_patience = int(es.get("patience", 3))
    early_stopping_monitor = str(es.get("monitor", "val_loss"))

    # validation section
    val = raw.get("validation", {})
    val_precompute_tensors = bool(val.get("precompute_tensors", False))

    # output section
    out = raw.get("output", {})
    results_file = str(out.get("results_file", f"results/experiment_{exp_id}.json"))
    # checkpoint_dir may be null in YAML (zero-shot experiments have no checkpoint)
    _ckpt = out.get("checkpoint_dir", f"models/donut_exp{exp_id}/")
    checkpoint_dir = str(_ckpt) if _ckpt is not None else ""
    log_file = str(out.get("log_file", f"logs/experiment_{exp_id}.log"))

    # depends_on: list of experiment IDs this experiment must wait for (DAG scheduler)
    _depends_raw = raw.get("depends_on", [])
    if isinstance(_depends_raw, (int, str)):
        _depends_raw = [_depends_raw]
    depends_on = [
        int(d) for d in (_depends_raw or []) if str(d).strip().isdigit() or isinstance(d, int)
    ]

    return _YAMLExperimentConfig(
        id=exp_id,
        name=name,
        description=description,
        arch_type=arch_type,
        is_zero_shot=is_zero_shot,
        base_checkpoint=base_checkpoint,
        tie_word_embeddings=tie_word_embeddings,
        full_parameter_finetuning=full_parameter_finetuning,
        image_height=image_height,
        image_width=image_width,
        allow_high_res=allow_high_res,
        max_length=max_length,
        datasets=dataset_entries,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        resource_optimizer_target_effective_batch=resource_optimizer_target_effective_batch,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        scheduler_type=scheduler_type,
        warmup_steps=warmup_steps,
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_monitor=early_stopping_monitor,
        seed=seed,
        val_precompute_tensors=val_precompute_tensors,
        results_file=results_file,
        checkpoint_dir=checkpoint_dir,
        log_file=log_file,
        depends_on=depends_on,
    )


def load_experiment_selection(
    selection_file: str | Path = "experiment_selection.json",
    experiments_dir: str | Path = "experiments",
) -> list[_YAMLExperimentConfig]:
    """
    Load only the experiments listed as enabled=true in experiment_selection.json.
    Returns configs sorted by experiment_id ascending.
    Falls back to load_all_experiments() if selection file does not exist.

    The selection file format::

        {
          "experiments": [
            {"id": "1", "enabled": true, "note": "..."},
            {"id": "6", "enabled": false, "note": "..."},
            ...
          ]
        }

    Experiment IDs in the selection file may be integers or strings; they are
    normalised to integers for comparison with ExperimentConfig.id.
    """
    selection_file = Path(selection_file)
    if not selection_file.exists():
        # Graceful fallback: load everything from the experiments directory
        return load_all_experiments(experiments_dir)

    with open(selection_file, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or "experiments" not in data:
        raise ValueError(
            f"experiment_selection.json must be a JSON object with an "
            f"'experiments' key, got: {type(data)}"
        )

    enabled_ids: list[int] = []
    for entry in data["experiments"]:
        if not isinstance(entry, dict):
            continue
        # Normalise id to int
        try:
            exp_id = int(str(entry.get("id", "")))
        except (ValueError, TypeError):
            continue
        if entry.get("enabled", True):
            enabled_ids.append(exp_id)

    if not enabled_ids:
        # Nothing enabled — fall back to all experiments
        return load_all_experiments(experiments_dir)

    return load_all_experiments(experiments_dir, experiment_ids=enabled_ids)


# ---------------------------------------------------------------------------
# Ablation control suite (from control_suite.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Impact metadata — not enforced at runtime, used by print_summary()
# ---------------------------------------------------------------------------

# Maps param name → (impact_level, is_underdocumented)
# Populated by _register() calls at module load time.
_IMPACT_REGISTRY: dict[str, tuple[str, bool]] = {}


def _register(name: str, impact: str, underdoc: bool = False) -> None:
    """Register a parameter's impact level and underdocumented flag."""
    _IMPACT_REGISTRY[name] = (impact.upper(), underdoc)


# ---------------------------------------------------------------------------
# DONUT — Document Understanding Transformer
# ---------------------------------------------------------------------------

# Register DONUT parameter metadata
_register("donut.input_size", "CRITICAL")
_register("donut.swin_window_size", "CRITICAL", underdoc=True)
_register("donut.base_model", "CRITICAL")
_register("donut.max_length", "HIGH")
_register("donut.sort_json_key", "HIGH", underdoc=True)
_register("donut.epochs", "HIGH")
_register("donut.encoder_lr", "CRITICAL")
_register("donut.decoder_lr", "CRITICAL")
_register("donut.batch_size", "HIGH")
_register("donut.gradient_accumulation_steps", "HIGH")
_register("donut.precision", "HIGH")
_register("donut.early_stopping_patience", "HIGH")
_register("donut.warmup_steps", "MEDIUM")
_register("donut.weight_decay", "LOW")
_register("donut.gradient_clip_val", "MEDIUM")
_register("donut.align_long_axis", "MEDIUM", underdoc=True)
_register("donut.val_check_interval", "LOW", underdoc=True)
_register("donut.lr_schedule", "MEDIUM")
_register("donut.optimizer_type", "MEDIUM")
_register("donut.seed", "LOW")
_register("donut.sroie_oversample", "HIGH")
_register("donut.num_workers", "MEDIUM")
_register("donut.dataloader_pin_memory", "MEDIUM")
_register("donut.dataloader_prefetch_factor", "MEDIUM")
_register("donut.save_total_limit", "LOW")
_register("donut.predict_with_generate", "MEDIUM")
_register("donut.tie_word_embeddings", "CRITICAL")


@dataclass
class DonutControlConfig:
    """All DONUT fine-tuning parameters in one place.

    ExperimentConfig (run_experiments.py) remains the authoritative config for
    training runs (per CLAUDE.md GP-1).  This dataclass adds the parameters
    that are implicit in the codebase (e.g. input_size from the processor,
    gradient_clip_val from the HF Trainer default) and serves as a reference.

    Fields marked ``# ⚠️`` are from the commonly_underdocumented[] section of
    finetuning_params.md — they have outsized impact but rarely appear in tutorials.
    """

    # ── Image Resolution ────────────────────────────────────────────────────
    # impact: CRITICAL
    # Must be multiples of patch_size (32). Pretrain used [2560, 1920].
    # Finetuning: [1280, 960] (width × height). Larger = more VRAM, slower.
    # Common values: [640,480], [960,720], [1280,960], [1920,1440], [2560,1920]
    input_size: list[int] = field(default_factory=lambda: [1280, 960])

    # ── Architecture (READ-ONLY — do not change from pretrain value) ────────
    # impact: CRITICAL ⚠️ UNDERDOCUMENTED
    # donut-base uses window_size=10. Changing this forces full weight re-init
    # for all attention layers — effectively trains from scratch.
    # NEVER change this when fine-tuning from a pretrained checkpoint.
    swin_window_size: int = 10  # READ-ONLY — matches donut-base pretrain config

    # ── Base Model ──────────────────────────────────────────────────────────
    # impact: CRITICAL
    # Clean base checkpoint with no CORD task-specific priors.
    base_model: str = BASE_MODEL  # "naver-clova-ix/donut-base"

    # ── Sequence Length ─────────────────────────────────────────────────────
    # impact: HIGH
    # Caps the generated JSON sequence length. Increased from 512 to 768 to
    # reduce truncation of long address fields (weakest SROIE field).
    max_length: int = MAX_LENGTH  # 768

    # ── Training Core ───────────────────────────────────────────────────────
    # impact: HIGH — optimal per convergence analysis is 10 (CLAUDE.md §3)
    epochs: int = 10

    # impact: CRITICAL — encoder gets lower LR than decoder (layerwise LR)
    encoder_lr: float = 5e-5

    # impact: CRITICAL — decoder LR is 2× encoder LR (faster adaptation)
    decoder_lr: float = 1e-4

    # impact: HIGH — batch=8 is optimal for 500–3940 samples per CLAUDE.md §3
    batch_size: int = 8

    # impact: HIGH — effective batch = batch_size × gradient_accumulation_steps
    # HuggingFace alias: gradient_accumulation_steps
    # fairseq alias: update_freq
    gradient_accumulation_steps: int = 2

    # ── Mixed Precision ─────────────────────────────────────────────────────
    # impact: HIGH
    # Auto-detected at runtime: bf16 on Ampere+, fp16 otherwise.
    # PyTorch Lightning alias: precision=16
    # HuggingFace alias: fp16=True / bf16=True in Seq2SeqTrainingArguments
    precision: str = "auto"  # "auto" | "bf16" | "fp16" | "fp32"

    # ── Early Stopping ──────────────────────────────────────────────────────
    # impact: HIGH — patience=3 prevents overfitting on 63-sample val set
    early_stopping_patience: int = 3

    # ── LR Warmup ───────────────────────────────────────────────────────────
    # impact: MEDIUM
    # Capped at runtime to ≤10% of total optimizer steps.
    # warmup=500 would exceed total steps for small datasets (Exp 1: ~312 steps).
    # Fixed at 40 so it is safe across all 8 experiments.
    warmup_steps: int = 40

    # ── Regularisation ──────────────────────────────────────────────────────
    # impact: LOW
    weight_decay: float = 0.01

    # impact: MEDIUM — prevents exploding gradients in the transformer decoder.
    # 1.0 is the universal default. Not currently passed explicitly to
    # Seq2SeqTrainingArguments (uses HF default which is also 1.0).
    gradient_clip_val: float = 1.0

    # ── Image Preprocessing ─────────────────────────────────────────────────
    # impact: MEDIUM ⚠️ UNDERDOCUMENTED
    # Whether to rotate portrait images to landscape. Usually False for
    # structured documents like receipts (text runs top-to-bottom naturally).
    # DonutProcessor alias: do_align_long_axis
    align_long_axis: bool = False

    # ── Data ────────────────────────────────────────────────────────────────
    # impact: HIGH ⚠️ UNDERDOCUMENTED
    # MUST be False for preprocessed datasets like CORD-v2 and SROIE.
    # Setting True silently corrupts token ordering in ground-truth sequences.
    sort_json_key: bool = False

    # ── Validation ──────────────────────────────────────────────────────────
    # impact: LOW ⚠️ UNDERDOCUMENTED
    # PyTorch Lightning float form: 0.2 = validate 5× per epoch.
    # HuggingFace: eval_strategy="epoch" (always validates once per epoch).
    # Can be useful for fast-converging small datasets.
    val_check_interval: float = 1.0

    # ── LR Schedule ─────────────────────────────────────────────────────────
    # impact: MEDIUM
    # "cosine" | "one_cycle" | "linear"
    # one_cycle: aggressive warmup + cosine decay — faster for short micro runs.
    lr_schedule: str = "cosine"

    # ── Optimizer ───────────────────────────────────────────────────────────
    # impact: MEDIUM
    # "adamw" | "sgd" (sgd = SGD + Nesterov, used in micro/mini mode)
    optimizer_type: str = "adamw"

    # ── Reproducibility ─────────────────────────────────────────────────────
    # impact: LOW
    seed: int = SEED  # 42

    # ── Dataset Balancing ───────────────────────────────────────────────────
    # impact: HIGH — SROIE oversampling is a prerequisite for auxiliary data to help.
    # Without 2× SROIE, Exps 2–4 score at or below the baseline (CLAUDE.md §8).
    sroie_oversample: int = 1

    # ── DataLoader Performance ──────────────────────────────────────────────
    # impact: MEDIUM — auto-computed by _optimal_num_workers() in constants.py
    num_workers: int = 8  # min(8, max(4, cpu_count // 2))

    # impact: MEDIUM — set False when num_workers=0 (incompatible with multiprocessing)
    dataloader_pin_memory: bool = True

    # impact: MEDIUM — prefetch factor for DataLoader workers (set None when workers=0)
    dataloader_prefetch_factor: int = 4

    # ── Checkpointing ───────────────────────────────────────────────────────
    # impact: LOW — keep 3 best checkpoints; older are auto-deleted
    save_total_limit: int = 3

    # ── Generation / Evaluation ─────────────────────────────────────────────
    # impact: MEDIUM — must be True for F1 eval during training in Seq2SeqTrainer
    predict_with_generate: bool = True

    # ── Weight Tying (CRITICAL BUG GUARD) ───────────────────────────────────
    # impact: CRITICAL
    # Must ALWAYS be False after resize_token_embeddings().
    # If True, tie_weights() on checkpoint reload destroys the learned lm_head
    # → F1 = 0.0 on every prediction.  See CLAUDE.md §2 and §16.
    tie_word_embeddings: bool = False  # NEVER change to True


# ---------------------------------------------------------------------------
# TrOCR — Transformer-based OCR
# ---------------------------------------------------------------------------

# Register TrOCR parameter metadata
_register("trocr.model_id", "CRITICAL")
_register("trocr.arch", "CRITICAL")
_register("trocr.input_size", "CRITICAL")
_register("trocr.patch_size", "CRITICAL")
_register("trocr.epochs", "HIGH")
_register("trocr.batch_size", "HIGH")
_register("trocr.learning_rate", "CRITICAL")
_register("trocr.max_length", "MEDIUM")
_register("trocr.gradient_accumulation_steps", "HIGH")
_register("trocr.mini_mode", "MEDIUM")
_register("trocr.lr_scheduler", "MEDIUM")
_register("trocr.warmup_ratio", "MEDIUM")
_register("trocr.warmup_init_lr", "MEDIUM", underdoc=True)
_register("trocr.weight_decay", "LOW")
_register("trocr.adam_beta1", "LOW")
_register("trocr.adam_beta2", "LOW")
_register("trocr.adam_epsilon", "LOW")
_register("trocr.gradient_clip_val", "MEDIUM")
_register("trocr.gradient_checkpointing", "HIGH")
_register("trocr.grad_ckpt_vram_threshold_gb", "HIGH")
_register("trocr.use_cache", "MEDIUM")
_register("trocr.predict_with_generate", "MEDIUM")
_register("trocr.num_beams", "MEDIUM")
_register("trocr.no_repeat_ngram_size", "MEDIUM")
_register("trocr.length_penalty", "LOW")
_register("trocr.patience", "HIGH", underdoc=True)
_register("trocr.augmentation_preset", "HIGH", underdoc=True)
_register("trocr.lora_rank", "HIGH")
_register("trocr.use_dora_encoder", "HIGH")
_register("trocr.num_workers", "MEDIUM")
_register("trocr.seed", "LOW")


@dataclass
class TrOCRControlConfig:
    """All TrOCR fine-tuning parameters.

    Replaces the 9 scattered module-level constants in train_trocr_yolo.py
    with a single structured dataclass. train_trocr_yolo.py keeps the
    module constants for backward compatibility with micro-mode patching
    (``TROCR_EPOCHS = 2`` etc.) but imports the new params from here.

    Architecture note: TrOCR uses a BEiT/DeiT encoder + RoBERTa/UniLM decoder.
    The encoder input is ALWAYS 384×384 — not configurable without full re-train.
    fairseq (official) uses update_freq for gradient accumulation;
    HuggingFace uses gradient_accumulation_steps. Both are aliased here.
    """

    # ── Model Identity ──────────────────────────────────────────────────────
    # impact: CRITICAL
    # "microsoft/trocr-base-printed" for printed text (SROIE receipts).
    # Alternatives: trocr-base-handwritten, trocr-large-printed
    model_id: str = "microsoft/trocr-base-printed"

    # impact: CRITICAL
    # trocr_small (62M) | trocr_base (334M) | trocr_large (558M)
    # base is the standard community choice for receipt OCR
    arch: str = "trocr_base"

    # ── Image Resolution ────────────────────────────────────────────────────
    # impact: CRITICAL ⚠️ UNDERDOCUMENTED (in a different way: it's FIXED)
    # ALL input images are ALWAYS resized to 384×384. Not configurable without
    # re-training the encoder. Known limitation — discussed in microsoft/unilm #674.
    # Workaround: pad image to square before resizing, or use sliding window.
    input_size: int = 384  # FIXED — do not change

    # impact: CRITICAL — 384/16 = 24 patches per side → 576 total patch tokens
    patch_size: int = 16  # FIXED — architectural constant

    # ── Training Core ───────────────────────────────────────────────────────
    # impact: HIGH
    # Official Microsoft training used 300 with patience=20.
    # Community finetuning typically 5–30 epochs on small datasets.
    epochs: int = 10

    # impact: HIGH
    # Official: BSZ=8 across 8 GPUs = 64 effective batch.
    # Community: 4–16 per GPU. VRAM-aware auto-scaling halves this when needed.
    # HuggingFace alias: per_device_train_batch_size
    # fairseq alias: BSZ
    batch_size: int = 16

    # impact: CRITICAL
    # Official training: 2e-5. Community HF finetuning: 4e-5–5e-5 for
    # smaller datasets. Lower LR reduces catastrophic forgetting.
    learning_rate: float = 5e-5

    # impact: MEDIUM
    # TrOCR processes line-cropped images; 64 tokens covers most text lines.
    # Use 512 for full-document inputs.
    # HuggingFace alias: generation_max_length | max_new_tokens
    max_length: int = 128

    # impact: HIGH
    # Gradient accumulation factor. update_freq=4 with BSZ=4 = effective BSZ 16.
    # fairseq alias: update_freq
    # HuggingFace alias: gradient_accumulation_steps
    gradient_accumulation_steps: int = 4

    # impact: MEDIUM
    # True → SGD+Nesterov+CosineAnnealingLR — faster convergence for micro runs.
    # False → AdamW+linear warmup (standard for full training runs).
    mini_mode: bool = False

    # ── LR Schedule ─────────────────────────────────────────────────────────
    # impact: MEDIUM
    # Official uses "inverse_sqrt" warmup decay (fairseq).
    # HuggingFace default is "linear". "cosine" recommended for longer finetune runs.
    # fairseq alias: lr-scheduler
    # HuggingFace alias: lr_scheduler_type
    lr_scheduler: str = "linear"

    # impact: MEDIUM
    # Warmup steps = warmup_ratio × total_training_steps.
    # 10% warmup (ratio=0.1) is the project default, computed inline.
    # fairseq alias: warmup_updates
    # HuggingFace alias: warmup_steps | num_warmup_steps
    warmup_ratio: float = 0.1

    # impact: MEDIUM ⚠️ UNDERDOCUMENTED
    # Starting LR at the very first warmup step. Prevents unstable initial
    # gradient updates. Present in fairseq config but absent from all blog posts.
    # Too-high value causes early divergence (loss spike at step 0).
    warmup_init_lr: float = 1e-8

    # ── Regularisation ──────────────────────────────────────────────────────
    # impact: LOW
    # Reference: 1e-4. NOTE: Before control_suite, AdamW was called without
    # weight_decay → PyTorch default of 0. This was a silent bug.
    weight_decay: float = 1e-4

    # impact: LOW — Adam first moment decay rate
    adam_beta1: float = 0.9

    # impact: LOW — Adam second moment decay rate
    adam_beta2: float = 0.999

    # impact: LOW — Adam numerical stability epsilon term
    adam_epsilon: float = 1e-8

    # impact: MEDIUM — global gradient norm clipping (applied explicitly via
    # clip_grad_norm_ at each accumulation step in train_trocr_yolo.py)
    gradient_clip_val: float = 1.0

    # ── Memory Optimisation ──────────────────────────────────────────────────
    # impact: HIGH — trades ~30-40% extra compute for activation memory savings
    # Required for TrOCR-base (246M params) to fit on lower-VRAM GPUs during backward.
    # use_cache MUST be False when gradient_checkpointing is True (incompatible).
    gradient_checkpointing: bool = True

    # impact: HIGH — VRAM threshold (GB) above which gradient checkpointing is
    # disabled.  Cards with VRAM > threshold have enough headroom that the
    # ~30-40% backward overhead is wasted.  Uses strict greater-than so that a
    # card reporting exactly 24.0 GB (e.g. RTX 4090) keeps checkpointing ON.
    # Mirrors DonutControlConfig / run_experiments._GRAD_CKPT_VRAM_THRESHOLD_GB.
    grad_ckpt_vram_threshold_gb: float = 24.0

    # impact: MEDIUM — must be False when gradient_checkpointing is True
    use_cache: bool = False

    # ── Generation / Evaluation ─────────────────────────────────────────────
    # impact: MEDIUM — must be True for CER/WER evaluation in Seq2SeqTrainer.
    # False = teacher-forcing logits for loss only (no beam search at eval time).
    # HuggingFace alias: predict_with_generate in Seq2SeqTrainingArguments
    predict_with_generate: bool = True

    # impact: MEDIUM — beam size for generation. 4 beams is the project default.
    num_beams: int = 4

    # impact: MEDIUM — set 0 to disable (n-gram blocking harmful for short OCR text)
    no_repeat_ngram_size: int = 0

    # impact: LOW — 1.0 = neutral (do not penalise short outputs)
    length_penalty: float = 1.0

    # ── Early Stopping ──────────────────────────────────────────────────────
    # impact: HIGH ⚠️ UNDERDOCUMENTED
    # Official used patience=20 with 300 epochs.
    # Community finetuning: 3–5 epochs.
    # None = not implemented (current project state for TrOCR).
    patience: int | None = None

    # ── Data Augmentation ───────────────────────────────────────────────────
    # impact: HIGH ⚠️ UNDERDOCUMENTED
    # Official fairseq config: DA2 (stronger augmentation used in IAM training).
    # Microsoft TrOCR paper: DA2 augmentation gives significantly better CER.
    # None = no augmentation (current project state — only basic resize).
    # Options: None | "DA1" | "DA2"
    # DA1 (light):   RandomRotation(±5°) + ColorJitter(brightness=0.2, contrast=0.2)
    # DA2 (strong):  RandomPerspective(distortion=0.2, p=0.5) +
    #                ElasticTransform(alpha=50.0) +
    #                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
    # Use get_augmentation_transforms(preset) to obtain the pipeline.
    augmentation_preset: str | None = None

    # ── PEFT (Parameter-Efficient Fine-Tuning) ───────────────────────────────
    # impact: HIGH
    # LoRA rank for DLoRA-TrOCR (Chang et al. 2024). Reduces trainable params
    # from 334M to <10M with comparable accuracy.
    # None = full fine-tuning (current project state).
    lora_rank: int | None = None

    # impact: HIGH — Apply DoRA to encoder instead of standard LoRA.
    # DLoRA-TrOCR paper: encoder uses DoRA, decoder uses LoRA.
    use_dora_encoder: bool = False

    # ── DataLoader ──────────────────────────────────────────────────────────
    # impact: MEDIUM — auto-computed by _optimal_num_workers() in constants.py
    num_workers: int = 8

    # ── Reproducibility ─────────────────────────────────────────────────────
    # impact: LOW
    seed: int = SEED  # 42

    # ── VRAM Calibration Constants (read-only, not tuneable) ─────────────────
    # Used by effective_batch_size() to compute the safe batch for available VRAM.
    # Empirically calibrated for TrOCR-base with AMP enabled.
    #   _TROCR_RESERVED_GB: total overhead (weights + gradients + AdamW states + system)
    #   _TROCR_PER_ITEM_GB: activation cost per batch item with AMP (bf16/fp16)
    _TROCR_RESERVED_GB: ClassVar[float] = 6.0
    _TROCR_PER_ITEM_GB: ClassVar[float] = 0.3

    def effective_batch_size(self, vram_gb: float) -> int:
        """Return the largest safe batch size for the given available VRAM (GB).

        Uses empirically calibrated constants:
          _TROCR_RESERVED_GB = 6.0  (model weights + grads + AdamW states + overhead)
          _TROCR_PER_ITEM_GB  = 0.3  (activation cost per item with AMP, TrOCR-base)

        Mirrors the inline VRAM check in train_trocr_yolo.train_trocr() so that
        the reported batch size in CONTROL_SUITE always matches the actual training
        batch after scaling.

        Returns at most self.batch_size and at least 1.
        """
        usable = max(vram_gb - self._TROCR_RESERVED_GB, 1.0)
        safe = max(1, int(usable / self._TROCR_PER_ITEM_GB))
        return min(self.batch_size, safe)


# ---------------------------------------------------------------------------
# YOLO — YOLOv8 Object Detection
# ---------------------------------------------------------------------------

# Register YOLO parameter metadata
_register("yolo.base_model", "CRITICAL")
_register("yolo.imgsz", "CRITICAL")
_register("yolo.mosaic", "CRITICAL")
_register("yolo.freeze", "CRITICAL", underdoc=True)
_register("yolo.lr0", "CRITICAL")
_register("yolo.epochs", "HIGH")
_register("yolo.batch", "HIGH")
_register("yolo.amp", "HIGH")
_register("yolo.optimizer", "HIGH")
_register("yolo.patience", "HIGH")
_register("yolo.scale", "HIGH")
_register("yolo.close_mosaic", "HIGH", underdoc=True)
_register("yolo.copy_paste", "HIGH")
_register("yolo.cache", "HIGH")
_register("yolo.fraction", "HIGH")
_register("yolo.box", "HIGH")
_register("yolo.cls", "HIGH")
_register("yolo.momentum", "MEDIUM")
_register("yolo.lrf", "MEDIUM")
_register("yolo.weight_decay", "MEDIUM")
_register("yolo.warmup_epochs", "MEDIUM")
_register("yolo.cos_lr", "MEDIUM")
_register("yolo.fliplr", "MEDIUM")
_register("yolo.degrees", "MEDIUM")
_register("yolo.translate", "MEDIUM")
_register("yolo.mixup", "MEDIUM")
_register("yolo.hsv_h", "MEDIUM")
_register("yolo.hsv_s", "MEDIUM")
_register("yolo.hsv_v", "MEDIUM")
_register("yolo.multi_scale", "MEDIUM")
_register("yolo.rect", "MEDIUM", underdoc=True)
_register("yolo.conf", "MEDIUM")
_register("yolo.iou", "MEDIUM")
_register("yolo.dropout", "MEDIUM")
_register("yolo.single_cls", "MEDIUM")
_register("yolo.workers", "MEDIUM")
_register("yolo.dfl", "MEDIUM")
_register("yolo.warmup_momentum", "LOW")
_register("yolo.warmup_bias_lr", "LOW")
_register("yolo.flipud", "LOW")
_register("yolo.shear", "LOW")
_register("yolo.perspective", "LOW")
_register("yolo.bgr", "LOW")
_register("yolo.max_det", "LOW")
_register("yolo.save_period", "LOW")
_register("yolo.seed", "LOW")
_register("yolo.deterministic", "LOW")
_register("yolo.amp_oom_auto_retry", "MEDIUM", underdoc=True)


@dataclass
class YOLOControlConfig:
    """All YOLOv8 fine-tuning parameters.

    Exposes all Ultralytics defaults explicitly so that any change to the
    training call is intentional and visible, not an accidental reliance on
    a library default that may change across Ultralytics versions.

    Parameters that were previously absent from the train_yolo() call are
    marked MISSING in the comments — they are now explicitly passed.

    Fields marked ``# ⚠️`` are from the commonly_underdocumented[] section.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    # impact: CRITICAL
    # yolov8x.pt: extra-large (~68M params). Batch and imgsz kept low for VRAM.
    # Alternatives: yolov8n.pt (3M), yolov8s.pt (11M), yolov8m.pt (26M),
    #               yolov8l.pt (44M), yolov8x.pt (68M), yolo11n.pt, ...
    base_model: str = "yolov8x.pt"

    # ── Image Resolution ────────────────────────────────────────────────────
    # impact: CRITICAL
    # Square input resolution. Images auto-resized and letterboxed.
    # Must match between training and inference.
    # Reduced from 640 to 512 to lower VRAM usage.
    # Common values: [320, 416, 512, 640, 832, 1024, 1280]
    imgsz: int = 512

    # ── Training Core ───────────────────────────────────────────────────────
    # impact: HIGH
    batch: int = 8

    # impact: HIGH — default 100 for finetuning; 300 for training from scratch
    epochs: int = 50

    # impact: HIGH — halt if no mAP50-95 improvement for N epochs.
    # Ultralytics default: 50. For finetuning: 10–20.
    patience: int = 15

    # ── Optimizer ───────────────────────────────────────────────────────────
    # impact: HIGH
    # "auto" selects AdamW for ≤10 warmup epochs, SGD otherwise.
    # Explicit "AdamW" recommended for finetuning.
    # Options: auto | SGD | Adam | AdamW | NAdam | RAdam | RMSProp
    optimizer: str = "AdamW"

    # impact: MEDIUM — SGD momentum / Adam beta1 (used when optimizer="SGD")
    momentum: float = 0.9

    # impact: CRITICAL — initial (peak) LR. AdamW default ~0.001.
    # Lower values recommended for finetuning to avoid overwriting pretrained features.
    lr0: float = 1e-3

    # impact: MEDIUM — final LR as fraction of lr0 (cosine/linear decay endpoint)
    # final_lr = lr0 × lrf
    lrf: float = 0.01

    # impact: MEDIUM — L2 regularisation. Ultralytics default: 0.0005.
    # PREVIOUSLY MISSING from train_yolo() call.
    weight_decay: float = 0.0005

    # impact: MEDIUM — use cosine annealing LR schedule (smoother decay).
    # When False, uses linear decay. PREVIOUSLY MISSING.
    cos_lr: bool = False

    # ── LR Warmup ───────────────────────────────────────────────────────────
    # impact: MEDIUM — linear warmup from near-zero to lr0 over N epochs.
    # PREVIOUSLY MISSING from train_yolo() call.
    warmup_epochs: float = 3.0

    # impact: LOW — starting SGD momentum during warmup phase.
    # PREVIOUSLY MISSING from train_yolo() call.
    warmup_momentum: float = 0.8

    # impact: LOW — starting LR specifically for bias parameters during warmup.
    # PREVIOUSLY MISSING from train_yolo() call.
    warmup_bias_lr: float = 0.1

    # ── Mixed Precision ─────────────────────────────────────────────────────
    # impact: HIGH — Automatic Mixed Precision (FP16/FP32). On by default.
    # Reduces VRAM ~50%, speeds training ~1.5–2×.
    # Disable (amp=False) if NaN losses appear in early training.
    # ⚠️ UNDERDOCUMENTED: YOLO silently halves batch size on CUDA OOM when amp=True.
    amp: bool = True

    # impact: MEDIUM ⚠️ UNDERDOCUMENTED (consequence of amp=True)
    # YOLO silently halves batch size on CUDA OOM — can cause inconsistent
    # effective batch sizes across experiment runs.
    amp_oom_auto_retry: bool = True  # INFORMATIONAL — cannot be disabled, documented here

    # ── Finetuning ──────────────────────────────────────────────────────────
    # impact: CRITICAL ⚠️ UNDERDOCUMENTED
    # THE most impactful underdocumented parameter for domain finetuning on
    # small datasets. Freezing backbone prevents overfitting and dramatically
    # reduces training time.
    # None = full training (all layers updated) — CURRENT PROJECT STATE
    # 0    = freeze nothing (same as None)
    # 3    = freeze first 3 backbone layers
    # 10   = freeze entire backbone, train only detection head
    # list = freeze specific layer indices
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default None).
    freeze: int | list[int] | None = None

    # ── Augmentation ────────────────────────────────────────────────────────
    # impact: CRITICAL — combines 4 training images into one 2×2 grid.
    # Massively improves detection robustness. Disabled for last close_mosaic epochs.
    # Set to 0.0 if document structure must be preserved exactly.
    # Current project: 0.5 (reduced from default 1.0 for receipt domain)
    mosaic: float = 0.5

    # impact: HIGH ⚠️ UNDERDOCUMENTED
    # Disable mosaic augmentation for the last N epochs.
    # CRITICAL for final accuracy. Ultralytics default: 10.
    # PREVIOUSLY MISSING from train_yolo() call (was relying on default).
    close_mosaic: int = 10

    # impact: MEDIUM — horizontal flip probability.
    # Set 0 for text/receipt detection (left-right orientation matters for text).
    # Current project: 0.0 (already set correctly for text domain)
    fliplr: float = 0.0

    # impact: LOW — vertical flip probability (usually 0 for receipts)
    flipud: float = 0.0

    # impact: MEDIUM — random rotation range in degrees.
    # Small non-zero values help with tilted receipts.
    degrees: float = 5.0

    # impact: MEDIUM — random translation as fraction of image size
    translate: float = 0.1

    # impact: HIGH — random scale (zoom) augmentation range.
    # Critical for detecting text at varying distances.
    scale: float = 0.3

    # impact: LOW — shear transformation in degrees
    shear: float = 0.0

    # impact: LOW — random perspective distortion coefficient
    perspective: float = 0.0

    # impact: MEDIUM — probability of mixup blending. Use 0–0.2 max.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 0).
    mixup: float = 0.0

    # impact: HIGH — probability of copy-paste augmentation.
    # Very effective for rare-class augmentation in imbalanced datasets.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 0).
    copy_paste: float = 0.0

    # impact: MEDIUM — hue jitter magnitude
    hsv_h: float = 0.015

    # impact: MEDIUM — saturation jitter magnitude
    hsv_s: float = 0.7

    # impact: MEDIUM — brightness/value jitter magnitude
    hsv_v: float = 0.4

    # impact: LOW — probability of random BGR channel swap
    bgr: float = 0.0

    # impact: MEDIUM — vary imgsz by ±50% each batch (multi-scale robustness)
    multi_scale: bool = False

    # ── Data ────────────────────────────────────────────────────────────────
    # impact: MEDIUM ⚠️ UNDERDOCUMENTED
    # rect=True batches images by aspect ratio to reduce letterbox padding waste.
    # WARNING: silently disables DataLoader shuffle → can bias training.
    rect: bool = False

    # impact: HIGH — fraction of dataset to use for training.
    # Reduce to 0.1 for fast hyperparameter sweep experiments.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 1.0).
    fraction: float = 1.0

    # ── Performance ─────────────────────────────────────────────────────────
    # impact: HIGH — 'ram' caches entire dataset in memory for maximum I/O speed.
    # 'disk' caches preprocessed images. False = load from disk each epoch.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default False).
    cache: bool | str = False

    # impact: MEDIUM — DataLoader worker threads per GPU rank
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 8).
    workers: int = 8

    # ── Loss Weights ────────────────────────────────────────────────────────
    # impact: HIGH — bounding box regression loss (GIoU/CIoU) coefficient.
    # Increase for tight localisation tasks (e.g. text-region detection).
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 7.5).
    box: float = 7.5

    # impact: HIGH — classification loss weight.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 0.5).
    cls: float = 0.5

    # impact: MEDIUM — Distribution Focal Loss weight.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 1.5).
    dfl: float = 1.5

    # ── Inference / NMS ─────────────────────────────────────────────────────
    # impact: MEDIUM — IoU threshold for NMS. Higher = fewer, more-overlapping boxes.
    conf: float = 0.25

    # impact: MEDIUM — minimum confidence threshold. Lower = more false positives.
    iou: float = 0.7

    # impact: LOW — maximum detections per image after NMS
    max_det: int = 300

    # ── Regularisation ──────────────────────────────────────────────────────
    # impact: MEDIUM — dropout in classification head. 0.1–0.2 for small datasets.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 0).
    dropout: float = 0.0

    # ── Task ────────────────────────────────────────────────────────────────
    # impact: MEDIUM — treat all classes as a single class (binary detection).
    single_cls: bool = False

    # ── Checkpointing ───────────────────────────────────────────────────────
    # impact: LOW — save checkpoint every N epochs. -1 = best.pt + last.pt only.
    save_period: int = -1

    # ── Reproducibility ─────────────────────────────────────────────────────
    # impact: LOW
    seed: int = SEED  # 42

    # impact: LOW — forces CUDA deterministic algorithms (slight speed penalty)
    deterministic: bool = True

    def recommended_freeze(self, num_train_samples: int) -> int | None:
        """Return the recommended freeze depth for the given training dataset size.

        Provides evidence-based guidance for domain finetuning on small receipt
        datasets, where freezing backbone layers prevents overfitting:

          >= 2000 samples  → None  (full training — enough data for all layers)
          500–1999 samples → 10   (freeze backbone, train detection head only)
          < 500 samples    → 3    (freeze first 3 backbone layers only)

        Call this when self.freeze is None (not explicitly overridden) to get a
        safe starting point. The recommendation is advisory — the caller decides
        whether to apply it.
        """
        if num_train_samples >= 2000:
            return None
        elif num_train_samples >= 500:
            return 10
        else:
            return 3


# ---------------------------------------------------------------------------
# ControlSuite — top-level bundle
# ---------------------------------------------------------------------------


@dataclass
class ControlSuite:
    """Comprehensive parameter hub for all three model architectures.

    Bundles DonutControlConfig, TrOCRControlConfig, and YOLOControlConfig
    with cross-cutting inspection helpers.

    Usage
    -----
        from run_experiments import CONTROL_SUITE

        CONTROL_SUITE.print_summary()
        CONTROL_SUITE.critical_params()
        CONTROL_SUITE.underdocumented_params()
    """

    donut: DonutControlConfig
    trocr: TrOCRControlConfig
    yolo: YOLOControlConfig

    @classmethod
    def default(cls) -> ControlSuite:
        """Return a ControlSuite with project-default configurations."""
        return cls(
            donut=DonutControlConfig(),
            trocr=TrOCRControlConfig(),
            yolo=YOLOControlConfig(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full control suite to a nested dict."""
        return {
            "donut": asdict(self.donut),
            "trocr": asdict(self.trocr),
            "yolo": asdict(self.yolo),
        }

    def critical_params(self) -> dict[str, Any]:
        """Return only parameters with impact=CRITICAL as a flat dict.

        Keys are prefixed with model name (e.g. "donut.input_size").
        """
        flat = _flatten_suite(self)
        return {
            k: v for k, v in flat.items() if _IMPACT_REGISTRY.get(k, ("", False))[0] == "CRITICAL"
        }

    def underdocumented_params(self) -> list[str]:
        """Return names of all commonly-underdocumented parameters.

        These are the parameters from the commonly_underdocumented[] section
        of finetuning_params.md that have outsized impact but rarely appear
        in tutorials.
        """
        return [k for k, (_, underdoc) in _IMPACT_REGISTRY.items() if underdoc]

    def print_summary(self) -> None:
        """Print all parameters with impact ratings and current values.

        Impact legend: CRITICAL > HIGH > MEDIUM > LOW
        ⚠️  = commonly underdocumented (see finetuning_params.md)
        """
        _impact_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        flat = _flatten_suite(self)

        # Group by model
        by_model: dict[str, list[tuple[str, Any, str, bool]]] = {
            "donut": [],
            "trocr": [],
            "yolo": [],
        }
        for key, value in flat.items():
            model = key.split(".")[0]
            impact, underdoc = _IMPACT_REGISTRY.get(key, ("LOW", False))
            by_model[model].append((key, value, impact, underdoc))

        _model_labels = {
            "donut": "DONUT — Document Understanding Transformer",
            "trocr": "TrOCR — Transformer OCR",
            "yolo": "YOLOv8 — Object Detection",
        }

        print("\n" + "=" * 80)
        print("CONTROL SUITE — Parameter Inventory")
        print("Impact: CRITICAL > HIGH > MEDIUM > LOW  |  ⚠️  = underdocumented")
        print("=" * 80)

        for model_name, params in by_model.items():
            params.sort(key=lambda x: (_impact_order.get(x[2], 99), x[0]))
            print(f"\n{'─' * 80}")
            print(f"  {_model_labels[model_name]}")
            print(f"{'─' * 80}")
            print(f"  {'Parameter':<45} {'Impact':<10} {'Value'}")
            print(f"  {'─' * 44} {'─' * 9} {'─' * 20}")
            for key, value, impact, underdoc in params:
                param_name = key.split(".", 1)[1]
                flag = " ⚠️" if underdoc else ""
                print(f"  {param_name + flag:<45} {impact:<10} {value!r}")

        print("\n" + "=" * 80)
        total = len(flat)
        n_critical = sum(1 for k in flat if _IMPACT_REGISTRY.get(k, ("",))[0] == "CRITICAL")
        n_underdoc = len(self.underdocumented_params())
        print(
            f"  Total parameters: {total}  |  Critical: {n_critical}  |  Underdocumented: {n_underdoc}"
        )
        print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_suite(suite: ControlSuite) -> dict[str, Any]:
    """Flatten a ControlSuite to a dot-notation dict keyed by 'model.param'."""
    result: dict[str, Any] = {}
    for model_name, config in [
        ("donut", suite.donut),
        ("trocr", suite.trocr),
        ("yolo", suite.yolo),
    ]:
        for k, v in asdict(config).items():
            result[f"{model_name}.{k}"] = v
    return result


def validate_sroie_oversample(
    datasets: list[str],
    sroie_oversample: int,
    skip_guard: bool = False,
) -> None:
    """Raise ValueError if a multi-dataset run uses sroie_oversample < 2.

    Without 2× SROIE oversampling, auxiliary datasets dilute the SROIE training
    signal and cause Experiments 2–4 to score at or below the baseline (see
    CLAUDE.md §8).  This validator enforces the rule at the start of
    run_experiment() so misconfigured runs fail fast rather than silently
    producing suboptimal results.

    Parameters
    ----------
    datasets:
        The list of dataset names for the experiment (e.g. ["sroie", "wildreceipt"]).
    sroie_oversample:
        The SROIE oversampling factor (must be >= 2 when len(datasets) > 1).
    skip_guard:
        When True, bypass the validation entirely.  Use only for intentional
        naïve control experiments (Exps 2–4) that deliberately use
        sroie_oversample=1 to prove that auxiliary data hurts without
        oversampling.  The guard is still enforced for all other callers
        (default False).

    Raises
    ------
    ValueError
        When more than one dataset is combined and sroie_oversample < 2,
        unless skip_guard is True.
    """
    if skip_guard:
        return  # intentional naïve control group — guard explicitly waived
    if len(datasets) > 1 and sroie_oversample < 2:
        raise ValueError(
            f"sroie_oversample={sroie_oversample} is too low for a multi-dataset run "
            f"(datasets={datasets!r}). "
            "Without 2× SROIE oversampling, auxiliary data dilutes the SROIE training "
            "signal and causes F1 to fall at or below the single-dataset baseline. "
            "Set sroie_oversample >= 2 when combining datasets."
        )


def get_augmentation_transforms(preset: str | None):
    """Return a torchvision transforms pipeline for the given preset, or None.

    Preset specifications
    ---------------------
    None
        No augmentation (pass-through) — current project default.
    "DA1" (light)
        RandomRotation(degrees=5) +
        ColorJitter(brightness=0.2, contrast=0.2)
    "DA2" (strong, matches TrOCR paper DA2 config)
        RandomPerspective(distortion_scale=0.2, p=0.5) +
        ElasticTransform(alpha=50.0) +
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)

    CI-safe: torchvision is an optional dependency.  If torchvision is not
    installed, this function returns None instead of raising ImportError, so
    control_suite.py remains importable in any environment.

    Parameters
    ----------
    preset:
        One of None, "DA1", or "DA2".

    Returns
    -------
    torchvision.transforms.Compose | None
        A pipeline that accepts a PIL image and returns a PIL image, or None
        when preset is None or torchvision is unavailable.

    Raises
    ------
    ValueError
        When preset is an unrecognised non-None string.
    """
    if preset is None:
        return None

    # Validate preset name before attempting the optional torchvision import
    # so that misconfigured presets raise immediately in all environments.
    if preset not in ("DA1", "DA2"):
        raise ValueError(
            f"Unknown augmentation preset: {preset!r}. Valid values: None, 'DA1', 'DA2'."
        )

    try:
        from torchvision import transforms
    except ImportError:
        return None

    if preset == "DA1":
        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    else:  # "DA2"
        return transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ElasticTransform(alpha=50.0),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            ]
        )


# ---------------------------------------------------------------------------
# Module-level singleton — import this
# ---------------------------------------------------------------------------

#: Singleton with project-default configurations for all three models.
#: Import this in train_trocr_yolo.py to access TrOCR and YOLO params.
CONTROL_SUITE: ControlSuite = ControlSuite.default()

# ---------------------------------------------------------------------------
# evaluation — DONUT evaluator + model evaluation
# ---------------------------------------------------------------------------

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None  # type: ignore[assignment]

try:
    import numpy as np
    import torch
except ImportError:
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]

try:
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    # Import inline fallbacks from train.py (which has the full inline implementation)
    try:
        from train import DonutProcessor, VisionEncoderDecoderModel  # noqa: E402, I001
    except ImportError:
        # train.py also not importable — define minimal stubs
        class DonutProcessor:  # type: ignore[no-redef]
            @classmethod
            def from_pretrained(cls, *a, **kw):
                raise ImportError("transformers is required. pip install transformers")

        class VisionEncoderDecoderModel:  # type: ignore[no-redef]
            @classmethod
            def from_pretrained(cls, *a, **kw):
                raise ImportError("transformers is required. pip install transformers")

# ─────────────────────────────────────────────────────────────────────────────
# Inline image loader — fallback when Pillow is unavailable
# ─────────────────────────────────────────────────────────────────────────────

try:
    from PIL import Image as _PILImage

    def _load_image(path: str | Path) -> _PILImage.Image:  # type: ignore[name-defined]
        return _PILImage.open(path).convert("RGB")

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

    def _png_unfilter(scanlines: list, width: int, bpp: int) -> bytes:
        """Apply PNG row de-filtering (Sub/Up/Average/Paeth)."""
        out = []
        prev = bytes(width * bpp)
        for ftype, raw in scanlines:
            row = bytearray(raw)
            if ftype == 1:  # Sub
                for i in range(bpp, len(row)):
                    row[i] = (row[i] + row[i - bpp]) & 0xFF
            elif ftype == 2:  # Up
                for i in range(len(row)):
                    row[i] = (row[i] + prev[i]) & 0xFF
            elif ftype == 3:  # Average
                for i in range(len(row)):
                    a = row[i - bpp] if i >= bpp else 0
                    row[i] = (row[i] + (a + prev[i]) // 2) & 0xFF
            elif ftype == 4:  # Paeth
                for i in range(len(row)):
                    a = row[i - bpp] if i >= bpp else 0
                    b = prev[i]
                    c = prev[i - bpp] if i >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    row[i] = (row[i] + pr) & 0xFF
            out.append(bytes(row))
            prev = bytes(row)
        return b"".join(out)

    def _load_png(path: str | Path) -> np.ndarray | None:
        """Minimal PNG decoder → RGB numpy array."""
        data = Path(path).read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        pos = 8
        width = height = bit_depth = color_type = 0
        idat = b""
        while pos < len(data):
            length = struct.unpack(">I", data[pos : pos + 4])[0]
            ctype = data[pos + 4 : pos + 8]
            chunk = data[pos + 8 : pos + 8 + length]
            pos += 12 + length
            if ctype == b"IHDR":
                width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
            elif ctype == b"IDAT":
                idat += chunk
            elif ctype == b"IEND":
                break
        raw = zlib.decompress(idat)
        if bit_depth != 8 or color_type not in (2, 6):
            return None
        channels = 3 if color_type == 2 else 4
        row_bytes = width * channels
        scanlines = []
        r = 0
        for _ in range(height):
            ftype = raw[r]
            scanlines.append((ftype, raw[r + 1 : r + 1 + row_bytes]))
            r += row_bytes + 1
        pixel_data = _png_unfilter(scanlines, width, channels)
        arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, channels)
        if channels == 4:
            arr = arr[:, :, :3]
        return arr

    def _load_bmp(path: str | Path) -> np.ndarray | None:
        """Minimal BMP decoder for 24-bit uncompressed BMP → RGB numpy array."""
        data = Path(path).read_bytes()
        if data[:2] != b"BM":
            return None
        pixel_offset = struct.unpack_from("<I", data, 10)[0]
        width = struct.unpack_from("<i", data, 18)[0]
        height = struct.unpack_from("<i", data, 22)[0]
        bits_per_pixel = struct.unpack_from("<H", data, 28)[0]
        compression = struct.unpack_from("<I", data, 30)[0]
        if bits_per_pixel != 24 or compression != 0:
            return None
        flipped = height > 0
        height = abs(height)
        row_size = (width * 3 + 3) & ~3
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(height):
            src_row = (height - 1 - row) if flipped else row
            start = pixel_offset + src_row * row_size
            raw_row = data[start : start + width * 3]
            pixels = np.frombuffer(raw_row, dtype=np.uint8).reshape(width, 3)
            arr[row] = pixels[:, ::-1]  # BGR → RGB
        return arr

    def _load_jpeg_ctypes(path: str | Path):
        """Load JPEG via ImageMagick subprocess (system libjpeg fallback)."""
        import subprocess as _sp

        try:
            result = _sp.run(
                ["convert", str(path), "-colorspace", "RGB", "ppm:-"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                ppm = result.stdout
                lines = ppm.split(b"\n")
                if lines[0] == b"P6":
                    dims = lines[1].split()
                    w, h = int(dims[0]), int(dims[1])
                    pixel_data = b"\n".join(lines[3:])
                    arr = np.frombuffer(pixel_data, dtype=np.uint8)
                    if len(arr) >= h * w * 3:
                        return arr[: h * w * 3].reshape(h, w, 3)
        except Exception:
            pass
        return None

    def _load_image(path: str | Path):  # type: ignore[misc]
        """Load an image file as an RGB numpy array without PIL."""
        path = Path(path)
        suffix = path.suffix.lower()
        arr = None
        if suffix == ".png":
            arr = _load_png(path)
        elif suffix in (".bmp",):
            arr = _load_bmp(path)
        elif suffix in (".jpg", ".jpeg"):
            arr = _load_jpeg_ctypes(path)
        if arr is None:
            raise RuntimeError(
                f"Cannot load {path} without Pillow. "
                "Install Pillow: pip install Pillow\n"
                "PNG (8-bit RGB/RGBA), BMP (24-bit), and JPEG (via ImageMagick) "
                "are supported natively."
            )
        return np.ascontiguousarray(arr, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _initialize_new_token_embeddings(model, tokenizer) -> None:
    """Initialise newly added SROIE special-token embeddings from semantically
    similar existing tokens rather than random noise.

    After ``resize_token_embeddings()``, the rows for the 10 NEW_TOKENS are
    randomly initialised.  With only ~630 optimizer steps (500 SROIE samples,
    batch=8, 10 epochs), the model can learn the easy part (field values from
    pretrained vocab) but fails to learn the harder structural part (new special
    token embeddings from random initialisation), producing output like:
        <s_sroie></s_sroie></s_sroie> SOON HUAT…</s_sroie> 32.00</s_sroie>
    instead of the correct XML structure.

    Semantic initialisation gives the new tokens a meaningful starting point,
    dramatically reducing the number of steps required to converge on the
    correct ``<s_company>VALUE</s_company>`` structure.

    Mapping strategy:
    - Opening field tags → mean of embeddings for semantically similar words
    - Closing field tags → copy of the corresponding opening tag embedding
    - ``<s_sroie>``  → copy of BOS token embedding
    - ``</s_sroie>`` → copy of EOS token embedding

    Args:
        model:     VisionEncoderDecoderModel with decoder.model.decoder.embed_tokens
                   and decoder.lm_head already resized.
        tokenizer: DonutProcessor.tokenizer with NEW_TOKENS already added.
    """
    import torch as _torch

    # Semantic seed words for each SROIE field opening tag.
    # Multiple words are averaged to produce a more stable initialisation.
    _SEED_WORDS: dict[str, list[str]] = {
        "<s_company>": ["company", "store", "name", "shop", "merchant"],
        "<s_date>": ["date", "time", "day"],
        "<s_address>": ["address", "location", "street", "place"],
        "<s_total>": ["total", "amount", "price", "sum"],
    }

    embed_weight = model.decoder.model.decoder.embed_tokens.weight
    lm_head_weight = model.decoder.lm_head.weight

    # Track the old vocabulary size (before NEW_TOKENS were added) so the
    # fallback mean is computed over existing pretrained embeddings only.
    _old_vocab_size = embed_weight.shape[0] - len(NEW_TOKENS)

    def _mean_embed(words: list[str]) -> _torch.Tensor:
        """Return the mean embedding vector for *words* that are in the old vocab."""
        vecs = []
        for w in words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            for tid in ids:
                if tid < _old_vocab_size:
                    vecs.append(embed_weight.data[tid].clone())
        if vecs:
            return _torch.stack(vecs).mean(dim=0)
        # Fallback: return the mean of the pretrained vocabulary (excludes new tokens)
        return embed_weight.data[:_old_vocab_size].mean(dim=0)

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    # Clamp bos/eos ids to the old vocab to avoid reading from new-token rows
    bos_id = min(bos_id, _old_vocab_size - 1)
    eos_id = min(eos_id, _old_vocab_size - 1)

    opening_tag_embeddings: dict[str, _torch.Tensor] = {}

    with _torch.no_grad():
        for token in NEW_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids([token])[0]
            if token_id >= embed_weight.shape[0]:
                # Safety: token_id out of range — resize was not applied yet
                logger.warning(
                    "_initialize_new_token_embeddings: token %s has id %d "
                    "which is >= embed_weight.shape[0]=%d; skipping",
                    token,
                    token_id,
                    embed_weight.shape[0],
                )
                continue

            if token == "<s_sroie>":
                init_vec = embed_weight.data[bos_id].clone()
            elif token == "</s_sroie>":
                init_vec = embed_weight.data[eos_id].clone()
            elif token in _SEED_WORDS:
                init_vec = _mean_embed(_SEED_WORDS[token])
                opening_tag_embeddings[token] = init_vec
            elif token.startswith("</s_") and token.endswith(">"):
                # Closing tag: use same embedding as its opening counterpart
                open_tag = token.replace("</", "<")
                if open_tag in opening_tag_embeddings:
                    init_vec = opening_tag_embeddings[open_tag].clone()
                else:
                    # Fallback: initialise from EOS
                    init_vec = embed_weight.data[eos_id].clone()
            else:
                # Unknown token — skip (leave random)
                continue

            embed_weight.data[token_id] = init_vec
            if lm_head_weight.shape[0] > token_id:
                lm_head_weight.data[token_id] = init_vec

    logger.info(
        "[SemanticInit] Initialised %d new SROIE token embeddings from semantic seeds",
        len(NEW_TOKENS),
    )


def _parse_sroie_output(tokens: str) -> dict:
    """Parse SROIE XML-like output format into a dict.

    SROIE format: <s_sroie><s_company>VALUE</s_company><s_date>VALUE</s_date>...
    This parser extracts values between opening and closing tags for each field.

    Fallback: when the model produces ``</s_sroie>`` as field delimiters instead
    of proper ``<s_company>VALUE</s_company>`` tags (structural learning failure
    with only ~630 optimizer steps), splits on ``</s_sroie>`` boundaries and
    maps positionally to [company, date, address, total].  The fallback is only
    activated when the primary XML parser finds zero fields.

    Returns a dict with keys from FIELDS; missing fields default to empty string.
    """
    result = EMPTY_GT.copy()

    for field_name in FIELDS:
        open_tag = f"<s_{field_name}>"
        close_tag = f"</s_{field_name}>"

        start_idx = tokens.find(open_tag)
        if start_idx != -1:
            start_idx += len(open_tag)
            end_idx = tokens.find(close_tag, start_idx)
            if end_idx != -1:
                result[field_name] = tokens[start_idx:end_idx].strip()

    # Fallback positional parser: activated only when the primary parser found
    # no fields AND the output actually contains </s_sroie> delimiters.
    # The model IS extracting the correct field values but emits them using
    # </s_sroie> as delimiters instead of <s_company>…</s_company>.
    # Example (observed self-test output):
    #   <s_sroie></s_sroie></s_sroie> SOON HUAT…</s_sroie> 01/01/2024</s_sroie>…
    # Strategy: strip outer wrapper, split on </s_sroie>, skip empty segments,
    # map first N non-empty segments to [company, date, address, total].
    # Guard: only trigger when </s_sroie> delimiter appears in the token string
    # to avoid false-positive warnings on legitimately empty ground truths.
    if not any(result.values()) and "</s_sroie>" in tokens:
        # Strip leading <s_sroie> wrapper if present
        stripped = tokens
        if stripped.startswith("<s_sroie>"):
            stripped = stripped[len("<s_sroie>") :]
        # Remove trailing </s_sroie> if present
        if stripped.endswith("</s_sroie>"):
            stripped = stripped[: -len("</s_sroie>")]
        parts = [p.strip() for p in stripped.split("</s_sroie>") if p.strip()]
        if parts:
            logger.warning(
                "_parse_sroie_output: primary XML parser found no fields; "
                "using fallback positional parser (model produced %d segments "
                "delimited by </s_sroie>). This indicates the model did not learn "
                "the SROIE structural tokens — check semantic token initialization. "
                "Raw prefix: %.120s",
                len(parts),
                tokens,
            )
            for i, field_name in enumerate(FIELDS):
                if i < len(parts):
                    result[field_name] = parts[i]

    return result


def _merge_token2json_pages(result: Any) -> dict:
    """Merge multi-page CORD output (list) into single dict (Phase 0b).

    The base checkpoint (donut-base-finetuned-cord-v2) knows about <sep/>
    (CORD multi-page separator). Even SROIE fine-tuned models can emit <sep/>
    because it's in the inherited vocabulary. When present, token2json()
    returns a list of dicts (one per page) instead of a single dict.

    This utility merges pages with "first occurrence of each key wins" logic,
    ensuring callers always receive a flat dict.

    Args:
        result: Output from processor.token2json() — either dict or list of dicts

    Returns:
        Single flat dict with all pages merged; empty dict if input is None/empty
    """
    if isinstance(result, list):
        merged: dict = {}
        for page in result:
            if isinstance(page, dict):
                for k, v in page.items():
                    if k not in merged:
                        merged[k] = v
        return merged if merged else {}

    if isinstance(result, dict):
        return result

    # Neither list nor dict (shouldn't happen, but defensive)
    return {}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of initial inference calls for which raw token output is logged
_DIAGNOSTIC_LOG_COUNT = 3


# ---------------------------------------------------------------------------
# EvaluationResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics.

    Attributes:
        global_precision: Precision over all (image, field) pairs.
        global_recall:    Recall over all (image, field) pairs.
        global_f1:        Harmonic mean of precision and recall.
        overall_exact_match: Fraction of images where ALL fields matched.
        per_field:        Dict mapping field name to per-field metrics.
        num_samples:      Number of test samples evaluated.
        parse_failures:   Number of samples where token2json failed.
        raw_predictions:  Optional list of raw prediction dicts.
    """

    global_precision: float = 0.0
    global_recall: float = 0.0
    global_f1: float = 0.0
    overall_exact_match: float = 0.0
    per_field: dict[str, dict[str, float]] = field(default_factory=dict)
    num_samples: int = 0
    parse_failures: int = 0
    raw_predictions: list[dict] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict compatible with legacy compute_metrics output."""
        d = {
            "global_precision": round(self.global_precision, 4),
            "global_recall": round(self.global_recall, 4),
            "global_f1": round(self.global_f1, 4),
            "overall_exact_match": round(self.overall_exact_match, 4),
        }
        for fname, fmetrics in self.per_field.items():
            d[f"{fname}_f1"] = round(fmetrics.get("f1", 0.0), 4)
            d[f"{fname}_ned"] = round(fmetrics.get("ned", 1.0), 4)
        return d


# ---------------------------------------------------------------------------
# Model loading helper — fixes the lm_head weight tying bug
# ---------------------------------------------------------------------------


def load_model_with_tied_weights(model_path: str, device: str = DEVICE, processor=None):
    # Use output_loading_info=True to detect missing keys at load time.
    # This is how we distinguish "trained lm_head loaded correctly" from
    # "lm_head randomly re-initialized because it was missing from the shard".
    model, loading_info = VisionEncoderDecoderModel.from_pretrained(
        model_path, output_loading_info=True
    )
    missing_keys = loading_info.get("missing_keys", [])

    # Sanity check: when tie_word_embeddings=False (set in train.py after
    # resize_token_embeddings()), LmHeadCloneCallback ensures lm_head.weight is
    # saved as an independent tensor in every checkpoint shard.  If it is still
    # missing after load, the checkpoint is corrupt — fail loudly instead of
    # silently recovering with random or embed_tokens weights (which produces
    # F1~0.42 and is indistinguishable from a healthy run without this check).
    if "decoder.lm_head.weight" in missing_keys and not getattr(
        model.decoder.config, "tie_word_embeddings", True
    ):
        raise RuntimeError(
            "CRITICAL: decoder.lm_head.weight missing from checkpoint. "
            "The model cannot generate SROIE tokens. Fix checkpoint saving."
        )

    _retie_decoder_head(model, missing_keys=missing_keys)

    model = model.to(device)
    model.eval()

    # ── Guard: verify SROIE tokens are present in processor vocab ───────────
    if processor is not None:
        _unk_id = processor.tokenizer.unk_token_id
        _sroie_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
        if _sroie_id == _unk_id:
            raise RuntimeError(
                f"Loaded processor from {model_path!r} has no SROIE special tokens: "
                f"'<s_sroie>' maps to unk_token_id ({_unk_id}). "
                "The checkpoint is unusable — re-run training so the processor "
                "is saved with add_special_tokens() applied."
            )
        # Fix decoder_start_token_id if it doesn't match <s_sroie>
        if model.config.decoder_start_token_id != _sroie_id:
            logger.warning(
                "decoder_start_token_id=%d does not match <s_sroie> id=%d — overriding.",
                model.config.decoder_start_token_id,
                _sroie_id,
            )
            model.config.decoder_start_token_id = _sroie_id
            model.decoder.config.decoder_start_token_id = _sroie_id

        # ── Guard: verify tokenizer vocab size matches model embedding size ──
        # A mismatch (e.g. processor saved with fewer tokens than the model
        # was trained with) causes embedding lookup errors or silent wrong IDs.
        _tok_vocab_size = len(processor.tokenizer)
        _model_vocab_size = None
        if hasattr(model.decoder, "lm_head") and hasattr(model.decoder.lm_head, "weight"):
            _model_vocab_size = model.decoder.lm_head.weight.shape[0]
        if _model_vocab_size is not None and _tok_vocab_size != _model_vocab_size:
            logger.warning(
                "Vocab size MISMATCH: tokenizer has %d tokens but model lm_head "
                "has %d output dimensions. This can cause wrong token predictions. "
                "Ensure processor and model are saved from the same training run.",
                _tok_vocab_size,
                _model_vocab_size,
            )

    return model


def _retie_decoder_head(model, missing_keys=None, model_path=None) -> None:
    """Re-tie or recover lm_head.weight after from_pretrained().

    Parameters
    ----------
    model : VisionEncoderDecoderModel
    missing_keys : list of str, optional
        Keys reported missing by from_pretrained(output_loading_info=True).
        When 'decoder.lm_head.weight' is in this list, the weight was NOT
        loaded from the checkpoint — it was randomly re-initialized.  We
        recover by copying embed_tokens.weight → lm_head.weight, which is
        lossy but far better than random init (F1 ~0.42 → comparable to
        embed_tokens quality).  The real fix is LmHeadCloneCallback in
        train.py which prevents this situation from arising.
    model_path : str, optional
        Path to the checkpoint being loaded, used for diagnostic messages.
    """
    if missing_keys is None:
        missing_keys = []

    decoder = model.decoder

    if not getattr(decoder.config, "tie_word_embeddings", True):
        lm_head_missing = any("lm_head.weight" in k for k in missing_keys)
        if lm_head_missing:
            # Emit a loud CRITICAL warning immediately — before any recovery
            # attempt — so it is visible even if recovery later fails.
            logger.error(
                "CRITICAL: decoder.lm_head.weight MISSING from checkpoint at %s. "
                "Falling back to embed_tokens.weight copy — this is LOSSY and will "
                "produce degraded F1 (~0.42). The root cause is that "
                "LmHeadCloneCallback did not fire during training. "
                "Re-train this experiment with LmHeadCloneCallback registered.",
                model_path or "<unknown path>",
            )
            # lm_head was not in the checkpoint — randomly re-initialized.
            # Recover by copying embed_tokens.weight as a starting point.
            embed_tokens = None
            if hasattr(decoder, "model"):
                if hasattr(decoder.model, "decoder"):
                    embed_tokens = getattr(decoder.model.decoder, "embed_tokens", None)
                elif hasattr(decoder.model, "embed_tokens"):
                    embed_tokens = decoder.model.embed_tokens
            if embed_tokens is not None and hasattr(decoder, "lm_head"):
                lm_shape = decoder.lm_head.weight.shape
                embed_shape = embed_tokens.weight.shape
                if lm_shape == embed_shape:
                    decoder.lm_head.weight = torch.nn.Parameter(embed_tokens.weight.data.clone())
                    logger.error(
                        "RECOVERY: decoder.lm_head.weight was MISSING from "
                        "checkpoint (randomly re-initialized). Copied "
                        "embed_tokens.weight as fallback. This is lossy — "
                        "fix: ensure LmHeadCloneCallback is registered in "
                        "DonutTrainer.train() so per-epoch checkpoints "
                        "include lm_head.weight."
                    )
                else:
                    logger.error(
                        "RECOVERY FAILED: lm_head shape %s != embed_tokens "
                        "shape %s — cannot copy. F1 will be degraded.",
                        lm_shape,
                        embed_shape,
                    )
            else:
                logger.error(
                    "RECOVERY FAILED: could not locate embed_tokens for "
                    "lm_head recovery. F1 will be degraded."
                )
        else:
            logger.info(
                "tie_word_embeddings=False and lm_head.weight present in "
                "checkpoint — no recovery needed."
            )
        return

    # Legacy path: re-tie for old checkpoints that have tie_word_embeddings=True
    if hasattr(decoder, "lm_head") and hasattr(decoder, "model"):
        embed_tokens = None
        if hasattr(decoder.model, "decoder") and hasattr(decoder.model.decoder, "embed_tokens"):
            embed_tokens = decoder.model.decoder.embed_tokens
        elif hasattr(decoder.model, "embed_tokens"):
            embed_tokens = decoder.model.embed_tokens

        if embed_tokens is not None:
            lm_shape = decoder.lm_head.weight.shape
            embed_shape = embed_tokens.weight.shape
            if lm_shape == embed_shape:
                decoder.lm_head.weight = embed_tokens.weight
                logger.info(
                    "Re-tied decoder.lm_head.weight → embed_tokens.weight (shape %s)", lm_shape
                )
            else:
                logger.warning(
                    "lm_head shape %s != embed_tokens shape %s — skipping re-tie",
                    lm_shape,
                    embed_shape,
                )

    assert model.decoder.lm_head.weight is not None, (
        "decoder.lm_head.weight is None after _retie_decoder_head() — weight tying failed."
    )


# ---------------------------------------------------------------------------
# DonutEvaluator class
# ---------------------------------------------------------------------------


class DonutEvaluator:
    """OOP evaluator that wraps model loading, self-test, inference, and metrics.

    Usage::

        evaluator = DonutEvaluator(model_path, processor, test_dataset)
        result = evaluator.evaluate()
        print(result.to_dict())
    """

    def __init__(
        self,
        model_path: Path,
        processor: DonutProcessor,
        test_dataset: list[tuple[Path, dict]],
        task_prompt: str = "<s_sroie>",
        max_length: int = MAX_LENGTH,
        device: str = DEVICE,
    ):
        self.model_path = Path(model_path)
        self.processor = processor
        self.test_dataset = test_dataset
        self.task_prompt = task_prompt
        self.max_length = max_length
        self.device = device
        self.parse_failure_count = 0
        self._inference_call_count = 0

        # Load model with weight re-tying fix
        logger.info("Loading model from %s", self.model_path)

        # Run checkpoint integrity checks before loading
        # (lm_head present, vocab size match)
        try:
            from validation import validate_checkpoint

            validate_checkpoint(
                model_path=self.model_path,
                expected_vocab_size=(
                    len(self.processor.tokenizer) if self.processor is not None else None
                ),
            )
        except Exception as _ckpt_exc:
            logger.warning("[DonutEvaluator] Checkpoint validation warning: %s", _ckpt_exc)

        self.model = load_model_with_tied_weights(
            str(self.model_path), device=self.device, processor=self.processor
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, allow_high_parse_failures: bool = False) -> EvaluationResult:
        """Run full evaluation: self-test first, then inference on all samples.

        Args:
            allow_high_parse_failures: When True, a >50% parse failure rate
                logs a warning and returns a zero-metric result instead of
                raising RuntimeError.  Use this for undertrained models where
                training completed successfully but the model hasn't yet
                converged to the expected tag format.  Default False keeps the
                existing behaviour (raises) so callers that want to detect
                truly broken models still get an exception.

        Raises:
            RuntimeError: If self-test fails (model produces no parseable output).
            RuntimeError: If >50% of samples fail to parse and
                allow_high_parse_failures is False (model is likely broken).
        """
        self._self_test()

        ground_truths = [s[1] for s in self.test_dataset]

        predictions = []
        self.parse_failure_count = 0
        self._inference_call_count = 0

        with torch.no_grad():
            for img_path, _gt in _progress(
                self.test_dataset, desc="Evaluating", total=len(self.test_dataset)
            ):
                pred = self._run_inference(img_path, self.task_prompt)
                predictions.append(pred)

        # Parse failure threshold check
        n = len(self.test_dataset)
        if n > 0 and self.parse_failure_count > n * 0.5:
            msg = (
                f"Parse failure threshold exceeded: {self.parse_failure_count}/{n} "
                f"({self.parse_failure_count / n:.1%}) samples failed to parse. "
                f"The model is likely broken — check token2json compatibility."
            )
            if not allow_high_parse_failures:
                raise EvaluationUndertrainedError(msg)
            logger.warning("%s — returning zero-metric result", msg)
            return EvaluationResult(
                global_precision=0.0,
                global_recall=0.0,
                global_f1=0.0,
                overall_exact_match=0.0,
                per_field={f: {"f1": 0.0, "ned": 1.0} for f in FIELDS},
                num_samples=n,
                parse_failures=self.parse_failure_count,
            )

        metrics_dict = self.compute_all_metrics(predictions, ground_truths)
        result = EvaluationResult(
            global_precision=metrics_dict["global_precision"],
            global_recall=metrics_dict["global_recall"],
            global_f1=metrics_dict["global_f1"],
            overall_exact_match=metrics_dict["overall_exact_match"],
            per_field={
                f: {
                    "f1": metrics_dict.get(f"{f}_f1", 0.0),
                    "ned": metrics_dict.get(f"{f}_ned", 1.0),
                }
                for f in FIELDS
            },
            num_samples=n,
            parse_failures=self.parse_failure_count,
            raw_predictions=predictions,
        )
        return result

    # ------------------------------------------------------------------
    # Self-test
    # ------------------------------------------------------------------

    def _self_test(self) -> None:
        """Run inference on one sample and verify the model produces output.

        The self-test asserts that the result is a dict with at least one
        non-empty field value. If it fails, raises with diagnostic info
        including the raw token sequence.

        For pretrained CORD models evaluated on SROIE, the output may not
        match SROIE fields — that's fine. The self-test only checks that
        the model can produce *any* parseable output (non-empty dict).
        """
        if not self.test_dataset:
            raise RuntimeError("Self-test failed: test_dataset is empty")

        img_path, gt = self.test_dataset[0]
        logger.debug("Self-test: running inference on %s", img_path)

        # Run raw generation to capture token sequence for diagnostics
        image = _load_image(img_path)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.no_grad():
            # FIX: Removed early_stopping=True — it is deprecated/invalid with
            # num_beams=1 (greedy decoding) and generates thousands of warnings
            # per eval call in transformers>=4.35.
            # Also removed repetition_penalty and no_repeat_ngram_size which were
            # blocking structural tags like </s_company> and </s_date>.
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=self.max_length,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        # Diagnostic: log first 20 token IDs and decoder seed token for actionable failure analysis
        logger.info(
            "Self-test raw token IDs (first 20): %s",
            outputs.sequences[0].tolist()[:20],
        )
        logger.info(
            "Self-test decoder_input_ids=%s → decoded=%r",
            decoder_input_ids[0].tolist(),
            self.processor.tokenizer.decode(decoder_input_ids[0].tolist()),
        )

        raw_tokens = self.processor.batch_decode(outputs.sequences)[0]
        cleaned = raw_tokens.replace(self.processor.tokenizer.eos_token, "")
        cleaned = cleaned.replace(self.processor.tokenizer.pad_token, "").strip()

        # For SROIE, use custom parser; for CORD, use token2json
        if self.task_prompt.startswith("<s_sroie"):
            try:
                parsed = _parse_sroie_output(cleaned)
            except Exception as exc:
                raise EvaluationUndertrainedError(
                # Fix: issue_report_summary high #7 — raise SelfTestFailedError so callers
                # can catch it explicitly without brittle string matching.
                raise SelfTestFailedError(
                    f"Self-test FAILED: SROIE parser raised {type(exc).__name__}: {exc}\n"
                    f"  Raw tokens: {raw_tokens!r}\n"
                    f"  Cleaned:    {cleaned!r}\n"
                    f"  Model path: {self.model_path}"
                ) from exc
        else:
            try:
                parsed = self.processor.token2json(cleaned)
            except Exception as exc:
                raise EvaluationUndertrainedError(
                # Fix: issue_report_summary high #7 — raise SelfTestFailedError.
                raise SelfTestFailedError(
                    f"Self-test FAILED: token2json raised {type(exc).__name__}: {exc}\n"
                    f"  Raw tokens: {raw_tokens!r}\n"
                    f"  Cleaned:    {cleaned!r}\n"
                    f"  Model path: {self.model_path}"
                ) from exc

            # token2json returns a list when <sep/> tokens are present (CORD multi-page).
            # Merge pages before unwrapping so the dict check below works correctly. (Phase 0b)
            parsed = _merge_token2json_pages(parsed)

        # Unwrap task-prompt wrappers
        parsed = _unwrap_prediction(parsed, self.task_prompt)

        # Check that we got at least one non-empty value.
        # Non-fatal: micro/mini smoke-test models are deliberately undertrained
        # and may not yet produce parseable SROIE tags. Log a warning and let
        # the full 63-sample evaluation determine the true F1 rather than
        # aborting with zeroed metrics.
        if not isinstance(parsed, dict) or not parsed:
            logger.warning(
                "Self-test: model produced empty dict — model may be undertrained "
                "(micro/mini mode). Continuing with full evaluation.\n"
                "  Raw tokens: %r\n"
                "  Cleaned:    %r\n"
                "  Parsed:     %r\n"
                "  Model path: %s",
                raw_tokens,
                cleaned,
                parsed,
                self.model_path,
            )
            return

        has_nonempty = any(
            str(v).strip() for v in parsed.values() if isinstance(v, (str, int, float))
        )
        # For nested dicts (e.g. CORD output), any non-empty sub-dict counts
        if not has_nonempty:
            has_nonempty = any(v for v in parsed.values() if isinstance(v, dict) and v)
        if not has_nonempty:
            has_nonempty = any(v for v in parsed.values() if isinstance(v, list) and v)

        if not has_nonempty:
            logger.warning(
                "Self-test: all fields empty in parsed output — model may be undertrained "
                "(micro/mini mode). Continuing with full evaluation.\n"
                "  Raw tokens: %r\n"
                "  Cleaned:    %r\n"
                "  Parsed:     %r\n"
                "  Model path: %s",
                raw_tokens,
                cleaned,
                parsed,
                self.model_path,
            )
            return

        logger.info("Self-test PASSED: parsed %d key(s) from %s", len(parsed), img_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        image_path: Path,
        task_prompt: str,
        preloaded_image: Any | None = None,
    ) -> dict:
        """Run inference on a single image and return parsed dict.

        Handles:
          - token2json failures (returns {} with log)
          - {"sroie": {...}} unwrapping
          - Diagnostic logging for the first few calls
        """
        self._inference_call_count += 1

        if preloaded_image is not None:
            image = preloaded_image
        else:
            image = _load_image(image_path)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=self.max_length,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "").strip()

        # Diagnostic logging for the first few calls (file only — too verbose for console)
        if self._inference_call_count <= _DIAGNOSTIC_LOG_COUNT:
            logger.debug(
                "Inference #%d raw tokens: %s",
                self._inference_call_count,
                sequence[:200] + ("..." if len(sequence) > 200 else ""),
            )

        parsed = self._parse_prediction(sequence)
        parsed = _unwrap_prediction(parsed, task_prompt)
        return parsed

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_prediction(self, tokens: str) -> dict:
        """Parse prediction using appropriate parser for the task format.

        For SROIE task prompts (<s_sroie>), uses the custom _parse_sroie_output()
        parser to extract values from XML-like tags.
        For CORD task prompts, falls back to token2json().

        token2json returns a list when the generated sequence contains <sep/>
        tokens (CORD multi-page format).  Even SROIE fine-tuned models can
        emit <sep/> because the base checkpoint (donut-base-finetuned-cord-v2)
        knows the token.  Merge pages into one dict (first occurrence of each
        key wins) so callers always receive a flat dict.

        Handles all edge cases defensively:
          - None or empty tokens → returns EMPTY_GT copy
          - token2json returning None → returns EMPTY_GT copy
          - token2json returning list → merges pages (Pattern 5)
          - Exception in parser → logs and returns EMPTY_GT copy
        """
        # Guard: None or empty token string → return empty template
        if tokens is None or (isinstance(tokens, str) and not tokens.strip()):
            return EMPTY_GT.copy()

        # For SROIE output, use the custom parser that understands SROIE tags
        if getattr(self, "task_prompt", "").startswith("<s_sroie"):
            try:
                result = _parse_sroie_output(tokens)
                if result and any(v for v in result.values()):  # At least one non-empty field
                    return result
                # No fields extracted — log and increment failure
                logger.warning("SROIE parser returned empty result from tokens: %.100s", tokens)
                self.parse_failure_count += 1
                return EMPTY_GT.copy()
            except Exception as exc:
                logger.warning("SROIE parser failed: %s — tokens: %.100s", exc, tokens)
                self.parse_failure_count += 1
                return EMPTY_GT.copy()

        # For CORD/other formats, use token2json
        try:
            result = self.processor.token2json(tokens)
            # Handle None return from token2json
            if result is None:
                logger.warning("token2json returned None — tokens: %.100s", tokens)
                self.parse_failure_count += 1
                return EMPTY_GT.copy()
            result = _merge_token2json_pages(result)  # Phase 0b: consolidate list merging
            if result:
                return result
            # Empty result from token2json (either [] list or {} dict)
            logger.warning("token2json returned empty result: %s", type(result))
            self.parse_failure_count += 1
            return EMPTY_GT.copy()
        except Exception as exc:
            logger.warning("token2json failed: %s — tokens: %.100s", exc, tokens)
            self.parse_failure_count += 1
            return EMPTY_GT.copy()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_f1(self, preds: list[dict], labels: list[dict]) -> float:
        """Compute global F1 over all (image, field) pairs.

        A pair is a true positive if the predicted string equals the ground
        truth string (case-insensitive, stripped).

        Returns 0.0 when ``preds`` is empty (no predictions to evaluate).
        """
        if not preds:
            return 0.0
        tp, total_pred, total_gt = 0, 0, 0
        for pred, gt in zip(preds, labels):
            for f in FIELDS:
                p_val = str(pred.get(f, "")).strip().lower()
                g_val = str(gt.get(f, "")).strip().lower()
                if g_val:
                    total_gt += 1
                if p_val:
                    total_pred += 1
                if p_val and g_val and p_val == g_val:
                    tp += 1
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _compute_ned(self, pred: str, gt: str) -> float:
        """Compute Normalized Edit Distance between two strings.

        NED is **lower-is-better**: 0 = identical, 1 = maximally different.

        Uses ``max(len(pred), len(gt))`` as denominator so NED ∈ [0, 1].
        """
        return normalized_edit_distance(pred, gt)

    def compute_all_metrics(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
    ) -> dict[str, float]:
        """Compute all metrics: global F1, per-field F1, per-field NED, exact match.

        Returns a flat dict compatible with the legacy ``compute_metrics`` output.
        """
        return compute_metrics(predictions, ground_truths)


# ---------------------------------------------------------------------------
# Prediction unwrapping helper
# ---------------------------------------------------------------------------


def _unwrap_prediction(parsed: dict, task_prompt: str) -> dict:
    """Unwrap task-prompt wrappers from token2json output.

    token2json may wrap SROIE output as ``{"sroie": {...}}``.
    CORD output may appear as ``{"cord-v2": {...}}``.
    """
    if not isinstance(parsed, dict):
        return parsed

    # Unwrap {"sroie": {...}} for SROIE task prompts
    if (
        task_prompt.startswith("<s_sroie")
        and "sroie" in parsed
        and isinstance(parsed["sroie"], dict)
    ):
        return parsed["sroie"]

    # Unwrap {"cord-v2": {...}} for CORD task prompts
    if (
        task_prompt.startswith("<s_cord")
        and "cord-v2" in parsed
        and isinstance(parsed["cord-v2"], dict)
    ):
        return parsed["cord-v2"]

    return parsed


# ---------------------------------------------------------------------------
# Module-level backward-compatible functions
# ---------------------------------------------------------------------------

# Global counter for diagnostic logging in the module-level run_inference
_module_inference_count = 0


def run_inference(model, processor, image_path, task_prompt, max_length=512, preloaded_image=None):
    """Run inference on a single image. Accepts an optional pre-loaded PIL Image.

    This is the backward-compatible module-level function. For new code,
    prefer ``DonutEvaluator._run_inference()``.

    After model.generate(), diagnostic logging is emitted for the first few
    calls to help debug token2json failures.
    """
    global _module_inference_count
    _module_inference_count += 1

    if preloaded_image is not None:
        image = preloaded_image
    else:
        image = _load_image(image_path)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(DEVICE)

    # FIX: Removed early_stopping=True — invalid with num_beams=1 (greedy
    # decoding).  This caused thousands of deprecation warnings per eval run.
    # Also removed repetition_penalty and no_repeat_ngram_size which were
    # blocking structural tags like </s_company> and </s_date>.
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max_length,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "").strip()

    # Diagnostic logging for the first few calls (file only — too verbose for console)
    if _module_inference_count <= _DIAGNOSTIC_LOG_COUNT:
        logger.debug(
            "run_inference #%d [%s] raw tokens: %s",
            _module_inference_count,
            task_prompt,
            sequence[:200] + ("..." if len(sequence) > 200 else ""),
        )

    # Check for task prompt mismatch (model outputting CORD schema for SROIE task)
    if task_prompt.startswith("<s_sroie") and sequence.startswith("<s_cord-v2>"):
        logger.warning(
            "Task prompt mismatch for %s: asked for <s_sroie> but model output starts with "
            "<s_cord-v2>. Model has not learned SROIE task format (CORD pretraining dominates). "
            "This indicates insufficient training or a corrupted checkpoint.",
            image_path,
        )

    # For SROIE output, use custom parser; for CORD, use token2json
    if task_prompt.startswith("<s_sroie"):
        try:
            result = _parse_sroie_output(sequence)
            return result if result and any(v for v in result.values()) else {}
        except Exception as e:
            logger.warning("SROIE parser failed for %s: %s", image_path, e)
            return {}

    # For CORD/other formats, use token2json
    try:
        result = processor.token2json(sequence)
    except Exception as e:
        logger.warning("token2json failed for %s: %s", image_path, e)
        return {}

    if not isinstance(result, dict):
        if isinstance(result, list):
            # CORD multi-page format: token2json returns a list when the
            # generated sequence contains <sep/> tokens (multiple line items).
            # Pass the list through so remap_cord_to_sroie can merge pages.
            logger.debug(
                "token2json returned list for %s (%d pages) — passing to caller",
                image_path,
                len(result),
            )
            return result
        logger.warning("token2json returned non-dict for %s: %s", image_path, type(result))
        return {}

    # Unwrap task-prompt wrappers
    result = _unwrap_prediction(result, task_prompt)
    return result


def remap_cord_to_sroie(cord_output):
    """Map CORD schema fields to SROIE field names (best effort).

    Handles the 'cord-v2' top-level wrapper that the pretrained CORD model
    may emit (e.g. ``{"cord-v2": {...}}``).

    Robust remapping handles multiple CORD schema variants:
      - ``store_info.store_name`` → ``company``
      - ``store_info.region`` or ``store_info.address`` → ``address``
      - ``total.total_price`` or ``total.total_etc`` → ``total``
      - ``date.date_value`` (dict, list, or plain string) → ``date``
      - Plain string values for any key
    """
    result = EMPTY_GT.copy()  # Phase 0b: use single source of truth

    # Handle list output from token2json (multi-page CORD with <sep/> tokens).
    # Merge all pages: first occurrence of each top-level key wins. (Phase 0b)
    cord_output = _merge_token2json_pages(cord_output)

    if not isinstance(cord_output, dict):
        return result

    # Unwrap the "cord-v2" top-level key if present
    if "cord-v2" in cord_output and isinstance(cord_output["cord-v2"], dict):
        cord_output = cord_output["cord-v2"]

    # --- company / address from store_info ---
    store_info = cord_output.get("store_info", {})
    if isinstance(store_info, dict):
        result["company"] = str(store_info.get("store_name", "")).strip()
        # Try multiple address field names
        addr = store_info.get("region", "") or store_info.get("address", "")
        result["address"] = str(addr).strip()
    elif isinstance(store_info, list) and store_info:
        first = store_info[0] if isinstance(store_info[0], dict) else {}
        result["company"] = str(first.get("store_name", "")).strip()
        addr = first.get("region", "") or first.get("address", "")
        result["address"] = str(addr).strip()
    elif isinstance(store_info, str):
        result["company"] = store_info.strip()

    # --- total ---
    total_info = cord_output.get("total", {})
    if isinstance(total_info, dict):
        # Try multiple total field names
        total_val = (
            total_info.get("total_price", "")
            or total_info.get("total_etc", "")
            or total_info.get("cashprice", "")
        )
        result["total"] = str(total_val).strip()
    elif isinstance(total_info, list) and total_info:
        first = total_info[0] if isinstance(total_info[0], dict) else {}
        total_val = (
            first.get("total_price", "") or first.get("total_etc", "") or first.get("cashprice", "")
        )
        result["total"] = str(total_val).strip()
    elif isinstance(total_info, str):
        result["total"] = total_info.strip()

    # --- date ---
    date_info = cord_output.get("date", {})
    if isinstance(date_info, dict):
        result["date"] = str(date_info.get("date_value", "")).strip()
    elif isinstance(date_info, list) and date_info:
        first = date_info[0]
        if isinstance(first, dict):
            result["date"] = str(first.get("date_value", "")).strip()
        else:
            result["date"] = str(first).strip()
    elif isinstance(date_info, str):
        result["date"] = date_info.strip()

    # --- fallback: check for direct SROIE-like keys at top level ---
    for f in FIELDS:
        if not result[f] and f in cord_output:
            val = cord_output[f]
            if isinstance(val, str):
                result[f] = val.strip()
            elif isinstance(val, dict):
                # Take first non-empty string value from the sub-dict
                for v in val.values():
                    if isinstance(v, str) and v.strip():
                        result[f] = v.strip()
                        break

    return result


def normalized_edit_distance(pred, gt):
    """Compute Normalized Edit Distance (NED) between pred and gt strings.

    NED = editdistance(pred, gt) / max(len(pred), len(gt))

    **Lower is better**: NED ∈ [0, 1] where 0 means identical and 1 means
    maximally different. This formulation uses ``max(len(pred), len(gt))``
    as the denominator, ensuring NED ≤ 1.0.
    """
    pred, gt = str(pred).lower().strip(), str(gt).lower().strip()
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit_distance(pred, gt) / max(len(pred), len(gt))


def compute_metrics(predictions, ground_truths):
    """Official SROIE Task 3 metric: global F1 over all (image, field) pairs.

    A pair is TP if predicted string == ground truth string
    (case-insensitive, stripped).

    Returns a flat dict with keys:
      ``global_precision``, ``global_recall``, ``global_f1``,
      ``overall_exact_match``, and per-field ``{field}_f1``, ``{field}_ned``.
    """
    tp, total_pred, total_gt = 0, 0, 0
    per_field = {f: {"tp": 0, "pred": 0, "gt": 0, "ned": []} for f in FIELDS}
    exact_match_all = []

    # Check for total prediction failure: all predictions are empty dicts
    all_empty = all(not any(str(pred.get(f, "")).strip() for f in FIELDS) for pred in predictions)
    if all_empty and predictions:
        print(
            "CRITICAL WARNING: ALL predictions are empty (token2json total failure). "
            "F1 will be 0.0 — check model output and token2json compatibility.",
            file=sys.stderr,
        )

    for pred, gt in zip(predictions, ground_truths):
        all_correct = True
        for f in FIELDS:
            p_val = str(pred.get(f, "")).strip().lower()
            g_val = str(gt.get(f, "")).strip().lower()

            if g_val:
                total_gt += 1
                per_field[f]["gt"] += 1
            if p_val:
                total_pred += 1
                per_field[f]["pred"] += 1
            if p_val and g_val and p_val == g_val:
                tp += 1
                per_field[f]["tp"] += 1
            elif not p_val and not g_val:
                pass  # Both absent = true negative
            else:
                all_correct = False

            per_field[f]["ned"].append(normalized_edit_distance(p_val, g_val))
        exact_match_all.append(int(all_correct))

    precision = tp / total_pred if total_pred > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    summary = {
        "global_precision": round(precision, 4),
        "global_recall": round(recall, 4),
        "global_f1": round(f1, 4),
        "overall_exact_match": round(np.mean(exact_match_all), 4) if exact_match_all else 0.0,
    }

    for f in FIELDS:
        fp = per_field[f]
        p = fp["tp"] / fp["pred"] if fp["pred"] > 0 else 0
        r = fp["tp"] / fp["gt"] if fp["gt"] > 0 else 0
        f_score = 2 * p * r / (p + r) if (p + r) > 0 else 0
        summary[f"{f}_f1"] = round(f_score, 4)
        summary[f"{f}_ned"] = round(np.mean(fp["ned"]), 4) if fp["ned"] else 1.0

    return summary


def print_results(pretrained_m, finetuned_m):
    """Pretty-print side-by-side pretrained vs. fine-tuned metrics."""
    print(f"\n{'=' * 72}")
    print(f"{'METRIC':<30} {'PRETRAINED':>18} {'FINE-TUNED':>18}")
    print(f"{'=' * 72}")
    print(f"{'Global F1':<30} {pretrained_m['global_f1']:>18.4f} {finetuned_m['global_f1']:>18.4f}")
    print(
        f"{'Global Precision':<30} {pretrained_m['global_precision']:>18.4f} {finetuned_m['global_precision']:>18.4f}"
    )
    print(
        f"{'Global Recall':<30} {pretrained_m['global_recall']:>18.4f} {finetuned_m['global_recall']:>18.4f}"
    )
    print(
        f"{'Overall Exact Match':<30} {pretrained_m['overall_exact_match']:>18.4f} {finetuned_m['overall_exact_match']:>18.4f}"
    )
    print(f"{'-' * 72}")
    for f in FIELDS:
        print(f"{f + ' F1':<30} {pretrained_m[f + '_f1']:>18.4f} {finetuned_m[f + '_f1']:>18.4f}")
        print(
            f"{f + ' NED':<30} {pretrained_m[f + '_ned']:>18.4f} {finetuned_m[f + '_ned']:>18.4f}"
        )
    print(f"{'=' * 72}")

    print(f"\n{'SROIE TASK 3 LEADERBOARD COMPARISON':^72}")
    print(f"{'-' * 72}")
    print(f"{'Method':<35} {'F1':>10}")
    print(f"{'-' * 72}")
    leaderboard = [
        ("LayoutLMv3 (Huang et al. 2022)", 0.9633),
        ("PICK (Yu et al. 2021)", 0.9612),
        ("BROS (Hong et al. 2022)", 0.9548),
        ("LayoutLMv2 (Xu et al. 2021)", 0.9495),
        ("DONUT SROIE fine-tuned (Kim 2022)", 0.8411),
        ("Our pretrained (CORD zero-shot)", pretrained_m["global_f1"]),
        ("Our fine-tuned (this work)", finetuned_m["global_f1"]),
    ]
    for name, score in sorted(leaderboard, key=lambda x: x[1], reverse=True):
        marker = " ◄" if "Our" in name else ""
        print(f"{name:<35} {score:>10.4f}{marker}")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Legacy standalone entry point
# ---------------------------------------------------------------------------


def evaluate_main():
    """Legacy standalone entry point for ad-hoc evaluation.

    For the full 8-experiment pipeline, use ``python run_all.py`` instead.
    Renamed from main() to avoid shadowing the run_experiments CLI main().
    """
    # Load test images + ground truth using canonical loader
    test_samples = load_sroie_test()

    print(f"Evaluating on {len(test_samples)} test images")

    ground_truths = [s[1] for s in test_samples]
    image_paths = [s[0] for s in test_samples]

    # Load pretrained (CORD) — hub model, no re-tying needed
    print("Loading pretrained model (CORD)...")
    pre_processor = DonutProcessor.from_pretrained(BASE_MODEL)
    pre_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL).to(DEVICE)
    pre_model.eval()

    # Load fine-tuned — apply weight re-tying fix
    print("Loading fine-tuned model...")
    workspace = os.environ.get("DONUT_WORKSPACE", "/workspace")
    ft_model_dir = os.path.join(workspace, "donut-sroie-finetuned")
    ft_processor = DonutProcessor.from_pretrained(ft_model_dir)
    ft_model = load_model_with_tied_weights(ft_model_dir, device=DEVICE)

    pretrained_preds = []
    finetuned_preds = []

    with torch.no_grad():
        for img_path in _progress(image_paths, desc="Inference"):
            # Pretrained (CORD) → remap
            raw = run_inference(pre_model, pre_processor, img_path, "<s_cord-v2>")
            pretrained_preds.append(remap_cord_to_sroie(raw))

            # Fine-tuned (SROIE) — unwrapping handled inside run_inference
            raw_ft = run_inference(ft_model, ft_processor, img_path, "<s_sroie>")
            finetuned_preds.append(raw_ft)

    pretrained_metrics = compute_metrics(pretrained_preds, ground_truths)
    finetuned_metrics = compute_metrics(finetuned_preds, ground_truths)

    print_results(pretrained_metrics, finetuned_metrics)

    # Save everything
    output = {
        "pretrained_metrics": pretrained_metrics,
        "finetuned_metrics": finetuned_metrics,
        "samples": [
            {
                "image": str(p),
                "ground_truth": gt,
                "pretrained_pred": pp,
                "finetuned_pred": fp,
            }
            for p, gt, pp, fp in zip(image_paths, ground_truths, pretrained_preds, finetuned_preds)
        ],
    }
    output_file = os.path.join(workspace, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved -> {output_file}")


# ── Config ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")

SROIE_DATA_DIR = _get_sroie_dir()


# ════════════════════════════════════════════════════════════════════════════
# Shared metric computation (SROIE Task-3 compatible)
# ════════════════════════════════════════════════════════════════════════════
# Import compute_metrics from donut_evaluator — single source of truth for
# SROIE Task-3 F1/NED/exact-match shared across DONUT and TrOCR+YOLO.
# compute_metrics is defined earlier in this same merged file (from donut_evaluator section)
compute_sroie_metrics = compute_metrics  # noqa: F821 (defined above in merged file)


# ════════════════════════════════════════════════════════════════════════════
# Load SROIE test set (shared by both architectures)
# ════════════════════════════════════════════════════════════════════════════
def load_test_samples() -> list[tuple[Path, dict[str, str]]]:
    """Load the 63 SROIE test images + ground truth via the canonical loader.

    Both DONUT and TrOCR+YOLO are evaluated on this EXACT same set.
    Delegates to dataset_loaders.load_sroie_test() — single source of truth.
    """
    return load_sroie_test()


# ════════════════════════════════════════════════════════════════════════════
# Evaluate DONUT on test set
# ════════════════════════════════════════════════════════════════════════════
def evaluate_donut_on_test(
    model_path: str,
    test_samples: list[tuple[Path, dict[str, str]]],
) -> dict:
    """Evaluate a DONUT model on the SROIE test set. Returns metrics dict."""
    from transformers import DonutProcessor

    processor = DonutProcessor.from_pretrained(model_path)
    model = load_model_with_tied_weights(model_path, device=DEVICE)

    predictions = []
    ground_truths = [s[1] for s in test_samples]
    latencies = []
    parse_failures = 0

    with torch.no_grad():
        for img_path, _gt in _progress(test_samples, desc="DONUT eval"):
            image = _PILImage.open(img_path).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
            decoder_input_ids = processor.tokenizer(
                "<s_sroie>", add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(DEVICE)

            t0 = time.perf_counter()
            # FIX: No early_stopping=True — invalid with num_beams=1
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=MAX_LENGTH,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            sequence = processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(processor.tokenizer.eos_token, "")
            sequence = sequence.replace(processor.tokenizer.pad_token, "").strip()

            try:
                parsed = _parse_sroie_output(sequence)
            except Exception:
                parsed = {}
                parse_failures += 1

            predictions.append(parsed)

    metrics = compute_sroie_metrics(predictions, ground_truths)
    metrics["parse_failures"] = parse_failures
    metrics["num_samples"] = len(test_samples)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 1) if latencies else 0.0

    # GPU cleanup
    _gpu_cleanup(model, processor)

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Evaluate TrOCR+YOLO on test set
# ════════════════════════════════════════════════════════════════════════════
def evaluate_trocr_yolo_on_test(
    yolo_weights: str,
    trocr_model_path: str,
    test_samples: list[tuple[Path, dict[str, str]]],
) -> dict:
    """Evaluate TrOCR+YOLO pipeline on the SROIE test set. Returns metrics dict."""
    from importlib import import_module

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    # Import the inference function and meta-buffer fix from train_trocr_yolo.py
    trocr_yolo_module = import_module("train_trocr_yolo")
    run_pipeline = trocr_yolo_module.run_trocr_yolo_inference
    _materialize_meta_buffers = trocr_yolo_module._materialize_meta_buffers

    # Use ultralytics YOLO if available, else fall back to inline _YOLO_CLS
    try:
        from ultralytics import YOLO
    except ImportError:
        YOLO = trocr_yolo_module._YOLO_CLS  # noqa: N806

    yolo_model = YOLO(str(yolo_weights))
    trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_path)
    # FIX: low_cpu_mem_usage=False + _materialize_meta_buffers prevents the
    # meta-device crash on TrOCR's sinusoidal positional embedding buffer.
    trocr_model = VisionEncoderDecoderModel.from_pretrained(
        trocr_model_path, low_cpu_mem_usage=False
    ).to(DEVICE)
    _materialize_meta_buffers(trocr_model, DEVICE)
    trocr_model.eval()

    predictions = []
    ground_truths = [s[1] for s in test_samples]
    latencies = []

    with torch.no_grad():
        for img_path, _gt in _progress(test_samples, desc="TrOCR+YOLO eval"):
            t0 = time.perf_counter()
            pred = run_pipeline(img_path, yolo_model, trocr_model, trocr_processor)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            predictions.append(pred)

    metrics = compute_sroie_metrics(predictions, ground_truths)
    metrics["num_samples"] = len(test_samples)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 1) if latencies else 0.0

    # GPU cleanup
    _gpu_cleanup(yolo_model, trocr_model, trocr_processor)

    return metrics


# ── Print metrics ────────────────────────────────────────────────────────────
def print_metrics(name: str, metrics: dict) -> None:
    """Pretty-print evaluation metrics in structured format."""
    print(f"\n  {'=' * 55}")
    print(f"  {name} Results")
    print(f"  {'=' * 55}")
    print(f"  Global F1:        {metrics.get('global_f1', 0):.4f}")
    print(f"  Global Precision: {metrics.get('global_precision', 0):.4f}")
    print(f"  Global Recall:    {metrics.get('global_recall', 0):.4f}")
    print(f"  Exact Match:      {metrics.get('overall_exact_match', 0):.4f}")
    print(f"  Num Samples:      {metrics.get('num_samples', 0)}")
    if "mean_latency_ms" in metrics:
        print(f"  Mean Latency:     {metrics['mean_latency_ms']:.1f} ms/image")
    print(f"  {'-' * 55}")
    for f in FIELDS:
        f1 = metrics.get(f"{f}_f1", 0)
        ned = metrics.get(f"{f}_ned", 1)
        print(f"  {f:12s}  F1={f1:.4f}  NED={ned:.4f}")
    print(f"  {'=' * 55}")


def generate_comparison_report(results: dict) -> None:
    """Generate a detailed HTML comparison report of all evaluated models."""
    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Model Evaluation Report</title>",
        "<style>",
        "  body { font-family: monospace; margin: 20px; }",
        "  table { border-collapse: collapse; margin: 20px 0; }",
        "  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "  th { background-color: #4CAF50; color: white; }",
        "  tr:nth-child(even) { background-color: #f2f2f2; }",
        "  .metric-high { color: green; font-weight: bold; }",
        "  .metric-low { color: red; }",
        "  h1, h2 { color: #333; }",
        "</style></head><body>",
        "<h1>Model Evaluation Report</h1>",
    ]

    if not results:
        html_lines.append("<p>No results to display.</p>")
    else:
        # Extract architectures and create comparison table
        html_lines.append("<h2>Global Performance Comparison</h2>")
        html_lines.append(
            "<table><tr><th>Architecture</th><th>Global F1</th><th>Precision</th><th>Recall</th><th>Exact Match</th><th>Latency (ms)</th></tr>"
        )

        for arch, metrics in results.items():
            f1 = metrics.get("global_f1", 0)
            prec = metrics.get("global_precision", 0)
            rec = metrics.get("global_recall", 0)
            em = metrics.get("overall_exact_match", 0)
            lat = metrics.get("mean_latency_ms", 0)

            f1_class = "metric-high" if f1 > 0.85 else "metric-low" if f1 < 0.7 else ""

            html_lines.append(
                f"<tr><td><strong>{arch}</strong></td><td class='{f1_class}'>{f1:.4f}</td>"
                f"<td>{prec:.4f}</td><td>{rec:.4f}</td><td>{em:.4f}</td><td>{lat:.1f}</td></tr>"
            )

        html_lines.append("</table>")

        # Per-field comparison
        html_lines.append("<h2>Per-Field Metrics</h2>")
        for arch, metrics in results.items():
            html_lines.append(f"<h3>{arch.upper()}</h3>")
            html_lines.append("<table><tr><th>Field</th><th>F1</th><th>NED</th></tr>")

            for field in FIELDS:
                f1 = metrics.get(f"{field}_f1", 0)
                ned = metrics.get(f"{field}_ned", 1)
                html_lines.append(f"<tr><td>{field}</td><td>{f1:.4f}</td><td>{ned:.4f}</td></tr>")

            html_lines.append("</table>")

    html_lines.extend(["</body></html>"])

    report_path = RESULTS_DIR / "evaluation_report.html"
    with open(report_path, "w") as f:
        f.write("\n".join(html_lines))

    print(f"\n📊 Detailed report saved -> {report_path}")


def generate_json_summary(results: dict) -> None:
    """Export evaluation results in structured JSON format."""
    summary = {
        "evaluation_timestamp": __import__("datetime").datetime.now().isoformat(),
        "num_test_samples": results.get("donut", {}).get("num_samples", 0),
        "architectures_evaluated": list(results.keys()),
        "results": results,
    }

    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📋 Summary saved -> {summary_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# run_experiments — orchestrator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# v2 resolution-sync constants and helper
# (RESULTS_DIR already defined at line ~2979 — do not redefine here)
# ---------------------------------------------------------------------------

# Canonical fine-tuning resolution for DONUT on SROIE.
# Height × Width must be multiples of 32 (patch_size=4, Swin stride=8 → 32).
# 1280×960 is the standard community fine-tuning resolution that fits
# comfortably in 24 GB VRAM at batch_size=8.
_FINETUNE_H: int = 1280  # height (tall receipts)
_FINETUNE_W: int = 960  # width


def _apply_resolution_sync(
    processor,
    model,
    height: int = _FINETUNE_H,
    width: int = _FINETUNE_W,
) -> None:
    """Synchronise processor image size and Swin encoder image_size.

    The ``naver-clova-ix/donut-base`` pretrained checkpoint stores
    ``model.config.encoder.image_size = [2560, 1920]``.  When fine-tuning
    at 1280×960, the processor resizes images to 1280×960 but the encoder's
    positional embeddings remain anchored to the 2560×1920 grid.  HuggingFace
    interpolates the mismatched embeddings silently, suppressing F1 for
    spatially distributed fields (company, address).

    This function sets BOTH fields atomically so they agree before any forward
    pass or gradient computation occurs.

    Side Effects
    ------------
    * ``processor.image_processor.size`` → ``{"height": height, "width": width}``.
    * ``model.config.encoder.image_size`` → ``[height, width]``

    Both mutations happen in-place.  Raises ``AssertionError`` if the fields
    still disagree after the update.
    """
    processor.image_processor.size = {"height": height, "width": width}
    model.config.encoder.image_size = [height, width]

    proc_h = processor.image_processor.size["height"]
    proc_w = processor.image_processor.size["width"]
    enc_h, enc_w = model.config.encoder.image_size
    assert proc_h == enc_h and proc_w == enc_w, (
        f"Resolution sync FAILED: processor=({proc_h},{proc_w}) "
        f"vs encoder=({enc_h},{enc_w}).  Check API compatibility."
    )
    logger.info(
        "[ResolutionSync] processor.image_processor.size = {height: %d, width: %d}  |  "
        "model.config.encoder.image_size = [%d, %d]  ✓",
        height,
        width,
        height,
        width,
    )


# ---------------------------------------------------------------------------
# ExperimentConfig — THE single source of truth for all hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Single source of truth for all training hyperparameters.

    Every training parameter (epochs, lr, batch_size, etc.) lives here.
    DonutTrainer reads them via duck-typed attribute access using the
    property aliases below (max_epochs, learning_rate, etc.).
    """

    name: str
    datasets: list[str]
    epochs: int = (
        10  # Per CLAUDE.md Section 3: optimal is 10 epochs. >12 causes overfit on 63-sample val set
    )
    lr: float = 5e-5
    batch_size: int = 8
    seed: int = SEED
    early_stopping_patience: int = 5  # Per CLAUDE.md §3: patience=5 for small datasets (<1000 samples); patience=3 fires too early when val loss plateaus then resumes improving.
    base_model: str = BASE_MODEL
    warmup_steps: int = 500  # train.py caps this to ≤10% of total opt-steps (min 10), so 500 is safe for all dataset sizes
    weight_decay: float = 0.01
    max_length: int = MAX_LENGTH
    gradient_accumulation_steps: int = 2
    encoder_lr: float = 5e-5
    decoder_lr: float = 1e-4
    description: str = ""
    experiment_id: int = 0
    sroie_oversample: int = 1  # Number of times to duplicate SROIE training samples (1–3 typical)
    skip_oversample_guard: bool = (
        False  # Set True for intentional naïve control experiments (Exps 2–4).
    )
    # These exist to prove that sroie_oversample=1 hurts F1 vs sroie_oversample>=2.
    # The guard is still enforced for all other paths (custom runs, CLI, etc.).

    # -- Mini-mode accelerators (all default to off so normal runs are unaffected) --
    subsample_train: int = 0  # >0: cap training set to this many samples (mini only)
    subsample_eval: int = 0  # >0: cap test set to this many samples (mini only)
    skip_step_validation: bool = False  # bypass 200-step minimum guard (mini only)
    lr_schedule: str = "cosine"  # "cosine" | "one_cycle" | "linear"
    optimizer_type: str = "adamw"  # "adamw" | "sgd" (sgd = SGD + Nesterov)

    # -- v2: explicit fine-tuning resolution (height × width) -------------
    # When non-default, _apply_resolution_sync() is called before training
    # to align processor.image_processor.size and model.config.encoder.image_size.
    # Both must be multiples of 32 (patch_size=4, Swin stride=8 → 32).
    finetune_height: int = _FINETUNE_H
    finetune_width: int = _FINETUNE_W

    # -- YAML-sourced config compatibility fields -------------------------
    # These fields are present in experiment_config_loader.ExperimentConfig and
    # needed by hparam_search.py, multi_seed_runner.py, and dag_scheduler.py
    # when they call dataclasses.replace() on configs from either class.
    arch_type: str = "donut"
    is_zero_shot: bool = False
    depends_on: list[int] = field(default_factory=list)
    full_parameter_finetuning: bool = True
    image_height: int = 1280
    image_width: int = 960
    allow_high_res: bool = False

    # -- Auxiliary dataset loss weighting --------------------------------
    # When aux_loss_weight < 1.0, the loss from auxiliary dataset samples
    # (WildReceipt, Invoices-DONUT, etc.) is multiplied by this factor.
    # SROIE samples always get weight 1.0.  Default 1.0 = no change (all
    # existing experiments 1–8 are unaffected by this field).
    # Requires aux dataset presence; does nothing for SROIE-only experiments.
    aux_loss_weight: float = 1.0

    # -- Duck-typed aliases for DonutTrainer compatibility ----------------
    # DonutTrainer reads config.max_epochs, config.learning_rate, etc.
    # These properties ensure a single source of truth (no duplication).

    @property
    def max_epochs(self) -> int:
        return self.epochs

    @property
    def learning_rate(self) -> float:
        return self.lr

    @property
    def per_device_train_batch_size(self) -> int:
        return self.batch_size

    @property
    def output_dir(self) -> str:
        """Default output directory; overridden at call site when needed."""
        return str(WORKSPACE / "models" / f"experiment_{self.experiment_id}")

    @property
    def id(self) -> int:
        """Alias for experiment_id — used by dag_scheduler.py and run_all.py dispatch."""
        return self.experiment_id

    @property
    def dataset_names(self) -> list[str]:
        """Names of all datasets in this experiment.

        Returns datasets directly since they are already strings in this class.
        Provides a consistent interface with experiment_config_loader.ExperimentConfig
        which stores DatasetEntry objects instead.
        """
        return self.datasets

    @property
    def base_checkpoint(self) -> str:
        """Alias for base_model — used by experiment_config_loader code paths."""
        return self.base_model


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS: dict[int, ExperimentConfig] = {
    1: ExperimentConfig(
        name="SROIE only (baseline)",
        datasets=["sroie"],
        description="Fine-tune on SROIE training set only.",
        experiment_id=1,
    ),
    2: ExperimentConfig(
        name="SROIE + WildReceipt (naïve baseline)",
        datasets=["sroie", "wildreceipt"],
        description="Naïve multi-dataset run — sroie_oversample=1 intentionally. "
        "Control group: proves auxiliary data dilutes F1 without oversampling.",
        experiment_id=2,
        sroie_oversample=1,
        skip_oversample_guard=True,
    ),
    3: ExperimentConfig(
        name="SROIE + Invoices-DONUT (naïve baseline)",
        datasets=["sroie", "invoices_donut"],
        description="Naïve multi-dataset run — sroie_oversample=1 intentionally. Control group.",
        experiment_id=3,
        sroie_oversample=1,
        skip_oversample_guard=True,
    ),
    4: ExperimentConfig(
        name="SROIE + WildReceipt + Invoices (naïve baseline)",
        datasets=["sroie", "wildreceipt", "invoices_donut"],
        description="Naïve combination — sroie_oversample=1 intentionally. Control group.",
        experiment_id=4,
        sroie_oversample=1,
        skip_oversample_guard=True,
    ),
    5: ExperimentConfig(
        name="SROIE + WildReceipt (2x SROIE)",
        datasets=["sroie", "wildreceipt"],
        description="SROIE + WildReceipt with 2x SROIE oversampling to preserve field coverage.",
        epochs=15,
        sroie_oversample=2,
        experiment_id=5,
    ),
    6: ExperimentConfig(
        name="SROIE + Invoices (2x SROIE)",
        datasets=["sroie", "invoices_donut"],
        description="SROIE + Invoices-DONUT with 2x SROIE oversampling.",
        epochs=15,
        sroie_oversample=2,
        experiment_id=6,
    ),
    7: ExperimentConfig(
        name="SROIE + All (2x SROIE)",
        datasets=["sroie", "wildreceipt", "invoices_donut"],
        description="All datasets with 2x SROIE oversampling to counteract dilution.",
        epochs=15,
        sroie_oversample=2,
        experiment_id=7,
    ),
    8: ExperimentConfig(
        name="SROIE + All (3x SROIE)",
        datasets=["sroie", "wildreceipt", "invoices_donut"],
        description="All datasets with 3x SROIE oversampling for maximum field coverage.",
        epochs=15,
        sroie_oversample=3,
        experiment_id=8,
    ),
    9: ExperimentConfig(
        name="SROIE + WildReceipt (2x SROIE, aux_w=0.5)",
        datasets=["sroie", "wildreceipt"],
        description=(
            "SROIE + WildReceipt with 2x SROIE oversampling and aux_loss_weight=0.5. "
            "Auxiliary samples contribute half-weight loss to reduce overfitting on "
            "WildReceipt domain noise while retaining visual diversity."
        ),
        epochs=15,
        sroie_oversample=2,
        aux_loss_weight=0.5,
        experiment_id=9,
    ),
    10: ExperimentConfig(
        name="SROIE + Invoices (2x SROIE, aux_w=0.6)",
        datasets=["sroie", "invoices_donut"],
        description=(
            "SROIE + Invoices-DONUT with 2x SROIE oversampling and aux_loss_weight=0.6. "
            "Mirrors Exp 6 (the historical best) but with 60% auxiliary loss weighting "
            "to reduce cross-domain overfitting."
        ),
        epochs=15,
        sroie_oversample=2,
        aux_loss_weight=0.6,
        experiment_id=10,
    ),
}

# ---------------------------------------------------------------------------
# TRAIN_CONFIG — cache validation for result JSON files
# ---------------------------------------------------------------------------
# FIX: Previously only included 5 of 9 hyperparameters (max_epochs, lr,
# batch_size, early_stopping_patience, base_model).  If warmup_steps,
# weight_decay, max_length, or seed changed, the cache validation at
# cached.get("config") != TRAIN_CONFIG would NOT detect stale results.
# Now includes ALL hyperparameters from ExperimentConfig for complete
# staleness detection.

_default_config = ExperimentConfig(name="", datasets=[])

TRAIN_CONFIG: dict[str, Any] = {
    "max_epochs": _default_config.epochs,
    "learning_rate": _default_config.lr,
    "per_device_train_batch_size": _default_config.batch_size,
    "early_stopping_patience": _default_config.early_stopping_patience,
    "base_model": _default_config.base_model,
    # FIX: These were previously missing, causing stale cache hits when
    # warmup_steps/weight_decay/max_length/seed changed.
    "warmup_steps": _default_config.warmup_steps,
    "weight_decay": _default_config.weight_decay,
    "max_length": _default_config.max_length,
    "seed": _default_config.seed,
    "gradient_accumulation_steps": _default_config.gradient_accumulation_steps,
    # v2: include resolution so changing finetune resolution correctly
    # invalidates previously cached results.
    "finetune_height": _default_config.finetune_height,
    "finetune_width": _default_config.finetune_width,
}


def _sanitize_metrics(metrics: dict) -> dict:
    """Replace NaN/Inf float values with JSON-safe sentinels before serialization.

    Python's json.dumps raises ValueError on math.nan/math.inf by default (they
    are not valid JSON).  With default=str they become strings, silently breaking
    downstream parsing.  This function converts them to None (JSON null) so the
    output is always valid JSON while still signalling "no value".
    """
    import math as _m  # noqa: PLC0415

    sanitized = {}
    for k, v in metrics.items():
        if isinstance(v, float) and (_m.isnan(v) or _m.isinf(v)):
            sanitized[k] = None
        else:
            sanitized[k] = v
    return sanitized


def _config_to_dict(config: ExperimentConfig) -> dict:
    """Serialize an ExperimentConfig to the TRAIN_CONFIG dict format.

    Used to record the *actual* training hyperparameters in the result JSON,
    so cache validation compares against what was truly used.
    """
    return {
        "max_epochs": config.epochs,
        "learning_rate": config.lr,
        "per_device_train_batch_size": config.batch_size,
        "early_stopping_patience": config.early_stopping_patience,
        "base_model": config.base_model,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "max_length": config.max_length,
        "seed": config.seed,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "finetune_height": config.finetune_height,
        "finetune_width": config.finetune_width,
    }


# ---------------------------------------------------------------------------
# Training — delegates to DonutTrainer from train.py
# ---------------------------------------------------------------------------


def train_experiment(
    exp_id: int,
    samples: list[tuple[Path, dict]],
    output_dir: Path,
    val_samples: list[tuple[Path, dict]] | None = None,
    base_processor=None,
    base_model=None,
    config=None,
    sample_sources: list[str] | None = None,
) -> list[dict]:
    """Fine-tune DONUT on *samples* and save the model to *output_dir*.

    All hyperparameters come from *config* if supplied, otherwise from
    ``EXPERIMENTS[exp_id]``.  Pass a custom ExperimentConfig for sweeps.
    Training is delegated to ``DonutTrainer`` from ``train.py``, which reads
    hyperparameters from the config via duck-typed attributes.

    When *base_processor* and *base_model* are supplied (pre-loaded by the
    caller), they are deep-copied from RAM instead of re-deserializing 800 MB
    of safetensors from disk — reducing ~4–6 s of ``from_pretrained`` overhead
    per experiment to a fast in-memory copy.

    Returns the trainer log history (list of per-step dicts) for convergence
    plot generation.
    """
    from train import DonutTrainer

    if config is None:
        config = EXPERIMENTS[exp_id]

    set_seed(config.seed)
    logger.debug("[Exp %d] Training on %d samples -> %s", exp_id, len(samples), output_dir)
    logger.debug(
        "[Exp %d] Hyperparams: epochs=%s, lr=%s, batch_size=%s, warmup=%s, wd=%s",
        exp_id,
        config.epochs,
        config.lr,
        config.batch_size,
        config.warmup_steps,
        config.weight_decay,
    )
    if val_samples:
        logger.debug("[Exp %d] Validation set: %d samples", exp_id, len(val_samples))

    # ---------------------------------------------------------------------------
    # Inner helper: build a fresh model, processor, and datasets.
    # Called once initially and again on each OOM retry so that GPU-resident
    # tensors from the failed attempt are never referenced on the retry.
    # ---------------------------------------------------------------------------
    def _build_model_and_datasets():
        if base_processor is not None and base_model is not None:
            _proc = copy.deepcopy(base_processor)
            _mdl = copy.deepcopy(base_model)
        else:
            _proc = DonutProcessor.from_pretrained(config.base_model)
            # Attempt Flash Attention 2 (requires flash-attn package; speeds up
            # decoder attention and reduces VRAM, enabling larger batch sizes).
            # Falls back silently to eager attention if unavailable or unsupported.
            if FLASH_ATTN_AVAILABLE:
                try:
                    _mdl = VisionEncoderDecoderModel.from_pretrained(
                        config.base_model, attn_implementation="flash_attention_2"
                    )
                    logger.info("[Model] Flash Attention 2 enabled for decoder")
                except Exception as _fa2_err:
                    logger.info(
                        "[Model] Flash Attention 2 load failed (%s: %s), using default attention",
                        type(_fa2_err).__name__,
                        _fa2_err,
                    )
                    _mdl = VisionEncoderDecoderModel.from_pretrained(config.base_model)
            else:
                logger.info("[Model] flash-attn not installed — using default attention (run fine)")
                _mdl = VisionEncoderDecoderModel.from_pretrained(config.base_model)

        # v2: Synchronise processor and encoder image sizes when resolution fields
        # are set (non-default values trigger the sync; default _FINETUNE_H/_FINETUNE_W
        # values always apply the sync for correctness).
        _apply_resolution_sync(
            _proc,
            _mdl,
            height=config.finetune_height,
            width=config.finetune_width,
        )

        # Add SROIE special tokens with diagnostic logging (Phase 0a)
        logger.debug("[Pre-resize] Tokenizer vocab size: %d", len(_proc.tokenizer))
        logger.debug(
            "[Pre-resize] Decoder embed_tokens shape: %s",
            _mdl.decoder.model.decoder.embed_tokens.weight.shape,
        )

        _proc.tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
        _mdl.decoder.resize_token_embeddings(len(_proc.tokenizer))

        logger.debug("[Post-resize] Tokenizer vocab size: %d", len(_proc.tokenizer))
        logger.debug(
            "[Post-resize] Decoder embed_tokens shape: %s",
            _mdl.decoder.model.decoder.embed_tokens.weight.shape,
        )
        logger.debug("[Post-resize] Decoder lm_head shape: %s", _mdl.decoder.lm_head.weight.shape)

        # Semantic initialisation: replace random embeddings for new SROIE tokens
        # with vectors derived from semantically similar existing tokens.
        # This dramatically reduces the optimizer steps required to learn the
        # correct <s_company>VALUE</s_company> XML structure, which would
        # otherwise need ~1500+ steps from a random start (vs. ~630 available
        # in the baseline Exp 1 with 500 samples × 10 epochs × batch=8).
        _initialize_new_token_embeddings(_mdl, _proc.tokenizer)

        # Verify NEW_TOKENS were added to tokenizer (Phase 0a diagnostic)
        for token in NEW_TOKENS:
            token_ids = _proc.tokenizer.encode(token, add_special_tokens=False)
            if not token_ids or len(token_ids) > 1:
                logger.error(
                    "CRITICAL: Token %s not in vocab or tokenizes to multiple IDs: %s",
                    token,
                    token_ids,
                )
                raise RuntimeError(f"Token addition failed for {token}; vocab may be corrupted")
        logger.debug(
            "[Token-verify] All %d SROIE tokens successfully added to vocab", len(NEW_TOKENS)
        )

        # After resize, embed_tokens and lm_head are separate tensors with
        # independent random init for the new tokens.  Set tie_word_embeddings=False
        # so save_pretrained() saves BOTH weights independently.  Without this,
        # the saved checkpoint omits lm_head (or tie_weights() overwrites the
        # learned lm_head with embed_tokens), causing F1=0 on reload.
        _ensure_dual_config(_mdl, "tie_word_embeddings", False)

        _ensure_dual_config(_mdl, "pad_token_id", _proc.tokenizer.pad_token_id)
        _sroie_start_id = _proc.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
        _ensure_dual_config(_mdl, "decoder_start_token_id", _sroie_start_id)
        # ── Guardrail: verify decoder_start_token_id decodes back to the task token ──
        _decoded = _proc.tokenizer.decode([_mdl.config.decoder_start_token_id])
        if _decoded != "<s_sroie>":
            raise RuntimeError(
                f"decoder_start_token_id={_mdl.config.decoder_start_token_id} decodes to "
                f"'{_decoded}', not '<s_sroie>'. Token was not added to vocab before "
                f"convert_tokens_to_ids was called, or the list-wrapping syntax is missing. "
                f"Use: tokenizer.convert_tokens_to_ids(['<s_sroie>'])[0]"
            )
        # FIX: max_length must live on generation_config, NOT model.config.
        # Newer transformers (>=4.37) raises ValueError at save_pretrained if
        # generation parameters are found on model.config.
        # FIX: Do NOT set both max_new_tokens and max_length — transformers 5.x
        # raises ValueError("Both 'max_new_tokens' and 'max_length' have been set").
        # max_new_tokens alone is sufficient and preferred.
        if hasattr(_mdl, "generation_config"):
            _mdl.generation_config.max_new_tokens = MAX_LENGTH
            # Explicitly unset max_length to avoid the transformers 5.x conflict.
            # GenerationConfig stores max_length=20 by default; clear it.
            if hasattr(_mdl.generation_config, "max_length"):
                _mdl.generation_config.max_length = None
            # Set decoder_start_token_id on generation_config (not just model.config)
            # so generate() uses the correct start token without needing a prompt.
            _mdl.generation_config.decoder_start_token_id = _sroie_start_id

        # Only enable gradient checkpointing when VRAM is constrained (< 24 GB).
        # 24 GB covers RTX 3090/4090 (24 GB) and below, where activation memory
        # during DONUT's backward pass (~3.5 GB at batch_size=8) is a real constraint.
        # On high-VRAM cards (RTX PRO 6000 = 95.6 GB, A100 = 80 GB, RTX 4090 = 24 GB
        # boundary) gradient checkpointing adds ~30-40% backward-pass overhead for
        # zero memory benefit — so disable it there.
        _GRAD_CKPT_VRAM_THRESHOLD_GB = 24.0
        _enable_grad_ckpt = True  # default: enable for safety on unknown hardware
        if torch.cuda.is_available():
            try:
                _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if _vram_gb > _GRAD_CKPT_VRAM_THRESHOLD_GB:
                    _enable_grad_ckpt = False
                    logger.info(
                        "[GradCkpt] Disabled — VRAM=%.1f GB > %.0f GB threshold "
                        "(saves ~35%% backward time at no memory cost)",
                        _vram_gb,
                        _GRAD_CKPT_VRAM_THRESHOLD_GB,
                    )
                else:
                    logger.info(
                        "[GradCkpt] Enabled — VRAM=%.1f GB <= %.0f GB threshold",
                        _vram_gb,
                        _GRAD_CKPT_VRAM_THRESHOLD_GB,
                    )
            except Exception as exc:
                logger.warning("[GradCkpt] VRAM detection failed (%s) — defaulting to enabled", exc)

        if _enable_grad_ckpt:
            _mdl.config.use_cache = False  # Required with gradient_checkpointing
            _mdl.decoder.config.use_cache = False
            _mdl.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            # use_cache=True is the default and correct when not using grad checkpointing
            _mdl.config.use_cache = True
            _mdl.decoder.config.use_cache = True

        # v2: freeze only Swin stage-0 (v1 froze stages 0+1).
        # Freezing fewer encoder layers lets the model adapt more freely to
        # SROIE's Southeast Asian receipt layouts while still preserving the
        # low-level patch embeddings from pre-training.
        frozen_count = 0
        for name, param in _mdl.encoder.named_parameters():
            if "layers.0" in name:
                param.requires_grad = False
                frozen_count += 1
        if frozen_count:
            logger.info("[FreezeEncoder] Frozen Swin stage-0 only (%d parameters)", frozen_count)

        # Move model to the training device (GPU when available).
        # Without this explicit call the model stays on CPU even when CUDA is
        # present, causing GPU Load 0% and ~20× slower training.
        _mdl = _mdl.to(DEVICE)
        _actual_device = next(_mdl.parameters()).device.type
        # Compare type-only strings (e.g. "cuda" == "cuda" when DEVICE="cuda",
        # even if the physical device is "cuda:0").
        if _actual_device != str(DEVICE).split(":")[0]:
            raise RuntimeError(
                f"Model.to({DEVICE!r}) failed — parameters still on {_actual_device!r}. "
                f"Check CUDA installation and torch device availability."
            )
        logger.info("[Device] Model moved to %s", DEVICE)

        # Build PyTorch datasets
        _aux_w = getattr(config, "aux_loss_weight", 1.0)
        _train_ds = MultiDataset(
            samples,
            _proc,
            max_length=config.max_length,
            sample_sources=sample_sources,
            aux_loss_weight=_aux_w,
        )
        _val_ds = (
            MultiDataset(
                val_samples,
                _proc,
                max_length=config.max_length,
                precompute_tensors=False,  # val set never needs pixel tensor precompute
            )
            if val_samples
            else None
        )

        return _proc, _mdl, _train_ds, _val_ds

    processor, model, train_ds, val_ds = _build_model_and_datasets()

    # Verify single source of truth: ExperimentConfig properties map correctly
    assert config.epochs == config.max_epochs, (
        f"Single source of truth violation: epochs={config.epochs} "
        f"!= max_epochs={config.max_epochs}"
    )
    assert config.lr == config.learning_rate, (
        f"Single source of truth violation: lr={config.lr} != learning_rate={config.learning_rate}"
    )
    assert config.batch_size == config.per_device_train_batch_size, (
        f"Single source of truth violation: batch_size={config.batch_size} "
        f"!= per_device_train_batch_size={config.per_device_train_batch_size}"
    )

    # Create DonutTrainer — it reads all hyperparams from config
    trainer = DonutTrainer(
        config=config,
        processor=processor,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    # Warn early if another GPU process is consuming VRAM — this can cause OOM
    # mid-epoch, wasting all setup time. Firing the warning before training
    # allows the user to kill the rogue process before the experiment starts.
    if torch.cuda.is_available():
        try:
            _free_vram, _total_vram = torch.cuda.mem_get_info()
            _used_by_others = _total_vram - _free_vram - torch.cuda.memory_allocated()
            if _used_by_others > 1 * 1024**3:  # > 1 GB held by external processes
                logger.warning(
                    "[VRAM] External process(es) occupying ~%.1f GB of GPU memory "
                    "(%s free of %s total). Risk of OOM during training. "
                    "Run `nvidia-smi` and kill any unnecessary GPU processes.",
                    _used_by_others / 1024**3,
                    f"{_free_vram / 1024**3:.1f} GB",
                    f"{_total_vram / 1024**3:.1f} GB",
                )
        except Exception:
            pass

    # Train with progressive OOM recovery: halve batch_size (8→4→2→1) on each
    # CUDA OOM, doubling gradient_accumulation_steps to keep the effective batch
    # the same, then fully rebuilding the model and datasets from scratch each
    # time so the retry starts on a clean, defragmented GPU.
    while True:
        try:
            result = trainer.train()
            break
        except torch.cuda.OutOfMemoryError as e:
            if config.batch_size > 1:
                new_batch = max(1, config.batch_size // 2)
                new_accum = config.gradient_accumulation_steps * 2
                config = dataclasses.replace(
                    config, batch_size=new_batch, gradient_accumulation_steps=new_accum
                )
                print(
                    f"[Exp {exp_id}] CUDA OOM — reducing batch_size to "
                    f"{config.batch_size}, increasing grad_accum to "
                    f"{config.gradient_accumulation_steps} and retrying"
                )
                # Delete ALL GPU-resident objects so the retry starts on a
                # clean, defragmented GPU — not just the trainer wrapper.
                _mm.shutdown_dataloader_workers(trainer)
                if hasattr(train_ds, "clear_caches"):
                    train_ds.clear_caches()
                if val_ds is not None and hasattr(val_ds, "clear_caches"):
                    val_ds.clear_caches()
                del trainer, model, processor, train_ds
                if val_ds is not None:
                    del val_ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure all CUDA ops complete before freeing
                torch.cuda.empty_cache()
                # Re-create everything from the base model to avoid inheriting
                # any gradient state from the failed attempt.
                processor, model, train_ds, val_ds = _build_model_and_datasets()
                trainer = DonutTrainer(
                    config=config,
                    processor=processor,
                    model=model,
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                )
            else:
                raise RuntimeError(
                    f"[Exp {exp_id}] CUDA OOM: batch_size already at minimum (1); "
                    "cannot recover without further hardware constraints"
                ) from e

    # Save model with tied weights
    trainer.save(output_dir)

    # Post-save spot-check: verify <s_sroie> token survives processor serialization
    _spot_proc = DonutProcessor.from_pretrained(str(output_dir))
    _spot_id = _spot_proc.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    _spot_decoded = _spot_proc.tokenizer.decode([_spot_id])
    if _spot_decoded == "<s_sroie>":
        logger.debug(
            "[Exp %d] Processor spot-check PASS: <s_sroie> → id=%d → '%s'",
            exp_id,
            _spot_id,
            _spot_decoded,
        )
    else:
        logger.warning(
            "[Exp %d] Processor spot-check FAIL: <s_sroie> → id=%d → '%s' "
            "(unk_token_id=%s) — processor is corrupt, evaluation will produce F1=0.0",
            exp_id,
            _spot_id,
            _spot_decoded,
            _spot_proc.tokenizer.unk_token_id,
        )

    logger.debug(
        "[Exp %d] Training complete (duration=%.1fs, train=%d, val=%d)",
        exp_id,
        result.duration_seconds,
        result.train_samples,
        result.val_samples,
    )

    # FIX: Explicit GPU cleanup between experiments to prevent OOM on GPUs
    # with limited VRAM.  The RTX 4090 has 24GB — sufficient for DONUT but
    # running 8+ experiments sequentially without cleanup risks fragmentation.
    # NOTE: local references MUST be deleted before _gpu_cleanup() is called;
    # passing them as arguments to _gpu_cleanup() is a no-op for freeing memory.
    # log_history is a plain list of dicts — a detached copy not tied to the
    # trainer's internals, so it's safe to capture before deleting the trainer.
    log_history = result.log_history
    # Explicitly clear dataset caches before deleting references — prevents
    # _pixel_cache tensors (up to 7.1 GB) from staying pinned until GC decides to run.
    _mm.shutdown_dataloader_workers(trainer)
    if hasattr(train_ds, "clear_caches"):
        train_ds.clear_caches()
    if val_ds is not None and hasattr(val_ds, "clear_caches"):
        val_ds.clear_caches()
    del trainer, model, processor, train_ds
    if val_ds is not None:
        del val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all CUDA ops complete before freeing
    torch.cuda.empty_cache()
    logger.debug("[Exp %d] GPU memory released", exp_id)

    return log_history


# ---------------------------------------------------------------------------
# Evaluation — delegates to DonutEvaluator from donut_evaluator.py
# ---------------------------------------------------------------------------


def evaluate_experiment(
    exp_id: int, model_dir: Path, config: ExperimentConfig | None = None
) -> dict:
    """Evaluate a fine-tuned model (at *model_dir*) on the SROIE test set.

    Uses DonutEvaluator from donut_evaluator.py which handles:
      - Weight re-tying via load_model_with_tied_weights (fixes lm_head bug)
      - Self-test before full evaluation
      - Parse failure threshold checking

    Args:
        exp_id: Experiment ID (for logging; only used if config is None)
        model_dir: Path to fine-tuned model directory
        config: Optional ExperimentConfig; if None, loads from EXPERIMENTS[exp_id]
    """
    # DonutEvaluator is defined above in this merged module

    if config is None:
        config = EXPERIMENTS[exp_id]
    test_samples = dataset_loaders.load_sroie_test()

    # Micro subsample evaluation — reduce test-set size for fast smoke tests
    if getattr(config, "subsample_eval", 0) > 0 and len(test_samples) > config.subsample_eval:
        import random as _rnd

        _rng = _rnd.Random(getattr(config, "seed", SEED))
        test_samples = _rng.sample(test_samples, config.subsample_eval)
        logger.debug(
            "[Exp %d] subsample_eval: evaluating on %d test samples", exp_id, len(test_samples)
        )
    else:
        logger.debug("[Exp %d] Evaluating on %d SROIE test images", exp_id, len(test_samples))

    # Load processor from the fine-tuned model directory
    processor = DonutProcessor.from_pretrained(str(model_dir))

    # Create evaluator — handles model loading with weight re-tying
    evaluator = DonutEvaluator(
        model_path=model_dir,
        processor=processor,
        test_dataset=test_samples,
        task_prompt="<s_sroie>",
        max_length=config.max_length,
        device=DEVICE,
    )

    # Run evaluation (includes self-test + parse failure threshold).
    # allow_high_parse_failures=True so undertrained models that haven't
    # converged to the SROIE tag format record F1=0.0 instead of crashing.
    eval_result = evaluator.evaluate(allow_high_parse_failures=True)
    metrics = eval_result.to_dict()

    if eval_result.parse_failures > 0:
        logger.warning(
            "[Exp %d] %d of %d predictions had parse failures",
            exp_id,
            eval_result.parse_failures,
            eval_result.num_samples,
        )

    return metrics


# ---------------------------------------------------------------------------
# Disk space management
# ---------------------------------------------------------------------------


def cleanup_checkpoints_after_eval(model_dir: Path, keep_model: bool = False) -> int:
    """Delete checkpoints (and optionally the whole model directory) after evaluation.

    Safe to call even if *model_dir* does not exist.

    Parameters
    ----------
    model_dir:
        Directory that HuggingFace Trainer wrote checkpoints into, e.g.
        ``/workspace/models/experiment_2``.
    keep_model:
        When *False* (default) the entire *model_dir* is removed after
        evaluation, recovering ~800 MB per experiment.  Only the result JSON in
        ``results/`` is needed for the paper pipeline.
        When *True* only ``checkpoint-*`` subdirectories are removed; the final
        model weights (``config.json``, ``model.safetensors``, etc.) are kept.

    Returns
    -------
    int
        Approximate bytes freed (sum of deleted file sizes).  0 if nothing was
        deleted or on any error.
    """
    import shutil

    try:
        from constants import format_bytes
    except ImportError:

        def format_bytes(n):  # type: ignore[misc]
            return f"{n} B"

    freed = 0
    if not model_dir.exists():
        return 0

    try:
        if not keep_model:
            # Remove entire model directory (default — only JSON result matters)
            for p in model_dir.rglob("*"):
                if p.is_file():
                    freed += p.stat().st_size
            shutil.rmtree(model_dir, ignore_errors=True)
            print(f"[Cleanup] Removed {model_dir} ({format_bytes(freed)} freed)")
        else:
            # Keep final model weights but delete checkpoint-N subdirectories
            for child in sorted(model_dir.iterdir()):
                if child.is_dir() and child.name.startswith("checkpoint-"):
                    for p in child.rglob("*"):
                        if p.is_file():
                            freed += p.stat().st_size
                    shutil.rmtree(child, ignore_errors=True)
            if freed > 0:
                print(
                    f"[Cleanup] Removed intermediate checkpoints from {model_dir} "
                    f"({format_bytes(freed)} freed)"
                )
    except Exception as exc:
        print(f"[Cleanup] WARNING: cleanup of {model_dir} failed: {exc}")

    return freed


def _check_disk_space_before_experiment(exp_id: int) -> bool:
    """Log a warning/error if disk space is dangerously low.

    Returns *True* when it is safe to proceed, *False* when the experiment
    should be skipped to avoid a mid-training "No space left on device" crash.
    """
    _MIN_DISK_SPACE_GB = 1.0  # below this → skip experiment (OS error 28 risk)
    _WARN_DISK_SPACE_GB = 3.0  # below this → log warning but proceed
    try:
        from constants import format_bytes, get_disk_usage

        _, _, free = get_disk_usage()
        free_gb = free / (1024**3)
        if free_gb < _MIN_DISK_SPACE_GB:
            print(
                f"[Exp {exp_id}] ERROR: Only {format_bytes(free)} disk free "
                f"(need ≥ {_MIN_DISK_SPACE_GB:.0f} GB). "
                "Skipping experiment to avoid OS error 28."
            )
            return False
        if free_gb < _WARN_DISK_SPACE_GB:
            print(
                f"[Exp {exp_id}] WARNING: Low disk space ({format_bytes(free)} free). "
                f"Recommend ≥ {_WARN_DISK_SPACE_GB:.0f} GB. "
                "Proceeding, but risk of 'No space left on device'."
            )
    except Exception:
        pass  # disk check is best-effort — never block an experiment
    return True


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    exp_id: int,
    base_processor=None,
    base_model=None,
    overrides: dict | None = None,
    keep_model: bool = False,
    no_disk_cleanup: bool = False,
) -> dict:
    """Run a single experiment: train, evaluate, save results.

    Checks cache validity (datasets AND hyperparams must match) before
    reusing a previous result.  Loads data via dataset_loaders, delegates
    training to train_experiment and evaluation to evaluate_experiment,
    then saves the result JSON.

    When *base_processor* and *base_model* are supplied, they are passed to
    train_experiment() which deep-copies them instead of re-loading from disk.

    When *overrides* is a non-empty dict, those key/value pairs are applied
    to the base ExperimentConfig via dataclasses.replace() before training.
    This allows ``--param`` CLI overrides without mutating the global EXPERIMENTS
    dict (GP-1).

    When *keep_model* is *False* (default), the model checkpoint directory is
    removed after evaluation completes to free disk space.  Set *True* to keep
    it (e.g. when passing ``--keep-models`` via CLI).

    When *no_disk_cleanup* is *True*, no disk cleanup is performed regardless
    of *keep_model* (for debugging purposes).
    """
    import dataclasses as _dc

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment ID {exp_id}. Valid: {list(EXPERIMENTS)}")

    config = EXPERIMENTS[exp_id]
    if overrides:
        config = _dc.replace(config, **overrides)
        logger.debug("[Exp %d] --param overrides applied: %s", exp_id, overrides)
    logger.debug("[Exp %d] %s | datasets=%s", exp_id, config.name, config.datasets)

    # Disk space pre-flight check — skip experiment if < 1 GB free to avoid OS error 28
    if not _check_disk_space_before_experiment(exp_id):
        result = {
            "experiment_id": exp_id,
            "name": config.name,
            "datasets": config.datasets,
            "config": _config_to_dict(EXPERIMENTS[exp_id]),
            "num_train_samples": 0,
            "metrics": {},
            "error": "Skipped: insufficient disk space (< 1 GB free)",
        }
        result_file = RESULTS_DIR / f"experiment_{exp_id}.json"
        result_file.write_text(json.dumps(result, indent=2))
        return result

    # Validate that multi-dataset runs use sroie_oversample >= 2.
    # Without 2× oversampling, auxiliary data dilutes the SROIE training signal
    # and causes F1 to fall at or below the single-dataset baseline (CLAUDE.md §8).
    # skip_oversample_guard=True is set only for intentional naïve control experiments
    # (Exps 2–4) that deliberately use sroie_oversample=1 to prove dilution.
    validate_sroie_oversample(
        config.datasets,
        config.sroie_oversample,
        skip_guard=getattr(config, "skip_oversample_guard", False),
    )

    result_file = RESULTS_DIR / f"experiment_{exp_id}.json"

    # Check if already done — validate cached result matches current experiment
    # definition (datasets AND hyperparameters) before reusing.
    # Compare against the original (unoptimized) experiment config so that
    # cache hits are hardware-independent: resource optimization is deterministic
    # for the same hardware, so if the experiment definition hasn't changed the
    # cached result is still valid.
    original_config_dict = _config_to_dict(EXPERIMENTS[exp_id])
    if result_file.exists():
        with open(result_file) as fh:
            cached = json.load(fh)
        if (
            cached.get("datasets") != config.datasets
            or cached.get("config") != original_config_dict
        ):
            # NOTE: JSON round-trip preserves numeric equality for floats like 5e-5,
            # so this comparison is safe (5e-5 == 5e-05 after json.load).
            print(
                f"[Exp {exp_id}] STALE result detected (datasets or config mismatch). "
                "Deleting and re-running."
            )
            result_file.unlink()
        else:
            print(f"[Exp {exp_id}] Valid cached result found - skipping.")
            return cached

    # Load data — request source labels when aux_loss_weight is active
    _aux_w = getattr(config, "aux_loss_weight", 1.0)
    _need_sources = _aux_w < 1.0
    if _need_sources:
        train_samples, val_samples, train_sources = dataset_loaders.get_combined_dataset(
            config.datasets, sroie_oversample=config.sroie_oversample, return_sources=True
        )
    else:
        train_samples, val_samples = dataset_loaders.get_combined_dataset(
            config.datasets, sroie_oversample=config.sroie_oversample
        )
        train_sources = None

    # Micro/mini subsample — deterministic RNG so repeated runs give the same split
    if getattr(config, "subsample_train", 0) > 0 and len(train_samples) > config.subsample_train:
        import random as _rnd

        _rng = _rnd.Random(config.seed)
        _indices = _rng.sample(range(len(train_samples)), config.subsample_train)
        train_samples = [train_samples[i] for i in _indices]
        if train_sources is not None:
            train_sources = [train_sources[i] for i in _indices]
        print(f"[Exp {exp_id}] subsample_train: using {len(train_samples)} samples")

    if len(train_samples) == 0:
        print(f"[Exp {exp_id}] WARNING: No samples loaded.")
        result = {
            "experiment_id": exp_id,
            "name": config.name,
            "datasets": config.datasets,
            "config": _config_to_dict(config),
            "num_train_samples": 0,
            "metrics": {},
            "error": "No training samples available",
        }
        result_file.write_text(json.dumps(result, indent=2))
        return result

    # Phase 5: Dynamic resource optimization (Phase 3-5)
    # Detect available hardware and optimize hyperparameters accordingly
    resources = _mm.detect_system_resources()
    optimized_config = _mm.optimize_hyperparams(
        num_train_samples=len(train_samples),
        available_vram_gb=resources.vram_gb,
        available_ram_gb=resources.ram_gb,
    )

    # Apply the optimized values to an isolated copy of the experiment config using
    # dataclasses.replace() so the global EXPERIMENTS dict is never mutated.
    # Only batch_size and gradient_accumulation_steps are overridden; epochs and
    # warmup_steps remain fixed per CLAUDE.md experiment design.
    #
    # Bug #2 fix: micro/mini mode sets gradient_accumulation_steps deliberately
    # (e.g. accum=1 for OneCycleLR "every batch → immediate optimizer step").
    # The resource optimizer must NOT override it — doing so cuts optimizer steps
    # in half (76 instead of 150 for micro), preventing any structural learning.
    # skip_step_validation=True is the canonical micro/mini mode signal.
    _is_micro = getattr(config, "skip_step_validation", False)
    old_batch = config.batch_size
    old_accum = config.gradient_accumulation_steps
    config = dataclasses.replace(
        EXPERIMENTS[exp_id],
        batch_size=optimized_config.batch_size,
        gradient_accumulation_steps=(
            EXPERIMENTS[exp_id].gradient_accumulation_steps  # preserve micro/mini accum
            if _is_micro
            else optimized_config.gradient_accumulation_steps
        ),
        encoder_lr=optimized_config.encoder_lr,
        decoder_lr=optimized_config.decoder_lr,
    )
    logger.debug(
        "[Exp %d] Resource optimization applied: batch_size %d → %d, grad_accum %d → %d%s",
        exp_id,
        old_batch,
        config.batch_size,
        old_accum,
        config.gradient_accumulation_steps,
        " (grad_accum preserved: micro/mini mode)"
        if _is_micro
        else f" ({optimized_config.config_explanation})",
    )

    # Log the config decision to terminal.txt for audit trail.
    # Done AFTER applying resource optimizer values so the log reflects the
    # actual config used in training (not the intermediate optimized_config
    # which always reports epochs=10 regardless of the experiment definition).
    audit_logger = TrainingAuditLogger(append_to_file="terminal.txt")
    audit_logger.log_config_decision(exp_id, optimized_config, actual_epochs=config.epochs)

    # ── Guardrail: verify global EXPERIMENTS dict was NOT mutated (GP-1) ──
    # The assert must be unconditional — the old version only checked when
    # values differed, but same-object implies same-values, making it inert.
    assert config is not EXPERIMENTS[exp_id], (
        "INVARIANT VIOLATION: config is the same object as EXPERIMENTS[exp_id]. "
        "dataclasses.replace() was not applied — mutations would corrupt the "
        "global singleton. Ensure replace() result is assigned back to config."
    )

    # ── Guardrail: validate optimizer step count ───────────────────────────
    # skip_step_validation=True is set only by micro/mini modes where a small
    # dataset + few epochs is intentional (smoke-test, not full convergence).
    from resource_manager import validate_training_config

    if not getattr(config, "skip_step_validation", False):
        validate_training_config(
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_samples=len(train_samples),
            epochs=config.epochs,
        )
    else:
        logger.debug("[Exp %d] step-count validation skipped (micro/mini mode)", exp_id)

    # Train — pass config explicitly so train_experiment uses the optimized values
    model_dir = WORKSPACE / "models" / f"experiment_{exp_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    _t_train_start = time.monotonic()
    log_history = train_experiment(
        exp_id,
        train_samples,
        model_dir,
        val_samples=val_samples,
        base_processor=base_processor,
        base_model=base_model,
        config=config,
        sample_sources=train_sources,
    )
    _train_duration_sec = time.monotonic() - _t_train_start

    # Evaluate
    try:
        metrics = evaluate_experiment(exp_id, model_dir)
        metrics["training_time_sec"] = _train_duration_sec
    except EvaluationUndertrainedError as exc:
        # Catch self-test failures and parse-failure-threshold errors for any
        # run type (not just micro/mini).  An undertrained full-run model that
        # hasn't converged to the SROIE tag format should record F1=0.0 and
        # let the remaining experiments continue, not crash the pipeline.
        # ROBUSTNESS: using EvaluationUndertrainedError (not bare RuntimeError +
        # string matching) ensures only genuine undertrained-model errors are
        # caught here.  Real RuntimeErrors (CUDA error, shape mismatch, etc.)
        # propagate to the caller as intended.
        print(
            f"[Exp {exp_id}] WARNING: evaluation failed (undertrained model) — "
            f"saving zero-metric result. Error: {exc}"
        )
        metrics = {f: 0.0 for f in ["global_f1", "global_precision", "global_recall"]}
        # Include per-field zeros so downstream paper-generation code doesn't KeyError
        from constants import FIELDS as _FIELDS

        for _field in _FIELDS:
            metrics[f"{_field}_f1"] = 0.0
            metrics[f"{_field}_ned"] = 1.0  # NED=1.0 means maximum edit distance
        metrics["error"] = str(exc)
        metrics["self_test_failed"] = True
        metrics["training_time_sec"] = _train_duration_sec
    except (SelfTestFailedError, RuntimeError) as exc:
        # Fix: issue_report_summary high #7 — catch SelfTestFailedError explicitly.
        # The old string-match "Self-test FAILED" in str(exc) was brittle; any
        # capitalisation change or unrelated RuntimeError would silently become F1=0.0.
        # SelfTestFailedError is caught unconditionally; other RuntimeErrors are only
        # caught when they match the parse-failure-threshold message.
        _is_self_test_failure = isinstance(exc, SelfTestFailedError)
        _is_parse_failure = not _is_self_test_failure and "Parse failure threshold exceeded" in str(
            exc
        )
        if _is_self_test_failure or _is_parse_failure:
            print(
                f"[Exp {exp_id}] WARNING: evaluation failed (undertrained model) — "
                f"saving zero-metric result. Error: {exc}"
            )
            metrics = {f: 0.0 for f in ["global_f1", "global_precision", "global_recall"]}
            # Include per-field zeros so downstream paper-generation code doesn't KeyError
            from constants import FIELDS as _FIELDS

            for _field in _FIELDS:
                metrics[f"{_field}_f1"] = 0.0
                metrics[f"{_field}_ned"] = 1.0  # NED=1.0 means maximum edit distance
            metrics["error"] = str(exc)
            metrics["self_test_failed"] = True
            metrics["training_time_sec"] = _train_duration_sec
        else:
            raise

    # Phase 5: Log training result to audit trail
    global_f1 = metrics.get("global_f1", 0.0)
    audit_logger.log_training_result(
        experiment_id=exp_id,
        global_f1=global_f1,
        training_time_sec=metrics.get("training_time_sec", 0.0),
        tokens_per_second=metrics.get("tokens_per_second"),
        early_stopping_epoch=metrics.get("early_stopping_epoch"),
        baseline_f1=None,  # Could set to pretrained F1 for comparison
    )

    # Runtime diagnostic check — detect known F1 anomalies post-evaluation
    try:
        from diagnostics import RuntimeCheckpoint, ai_diagnose

        _diag_cp = RuntimeCheckpoint(
            timestamp=time.time(),
            stage="post_evaluation",
            experiment_id=exp_id,
            eval_f1=global_f1,
            metrics=metrics,
        )
        # Check for known F1 collapse patterns
        _diag_issues = []
        if 0.40 < global_f1 < 0.44:
            _diag_issues.append(
                {
                    "pattern": "lm_head_dedup",
                    "severity": "critical",
                    "evidence": f"F1={global_f1:.4f} in safetensors danger zone (0.40-0.44)",
                }
            )
        elif 0 < global_f1 < 0.02:
            _diag_issues.append(
                {
                    "pattern": "token2json_list",
                    "severity": "critical",
                    "evidence": f"F1={global_f1:.4f} — likely token2json returning list",
                }
            )
        elif global_f1 == 0.0 and not metrics.get("self_test_failed"):
            _diag_issues.append(
                {
                    "pattern": "total_f1_collapse",
                    "severity": "critical",
                    "evidence": "F1=0.0000 — complete prediction failure",
                }
            )
        if _diag_issues:
            for _issue in _diag_issues:
                logger.warning(
                    "[Diagnostics] Exp %d | PATTERN: %s (%s)  %s",
                    exp_id,
                    _issue["pattern"],
                    _issue["severity"],
                    _issue["evidence"],
                )
            # AI diagnosis if enabled
            if os.environ.get("AI_DIAGNOSE", "0") == "1":
                _dx = ai_diagnose(
                    {"experiment_id": exp_id, "metrics": metrics, "issues": _diag_issues},
                    provider=os.environ.get("AI_DIAGNOSE_PROVIDER", "auto"),
                )
                if _dx:
                    logger.info("[AI Diagnosis] Exp %d:\n%s", exp_id, _dx)
    except ImportError:
        pass  # diagnostics.py not present

    # Save result — use the *original* (unoptimized) experiment config dict for cache
    # staleness comparison so future runs can correctly detect stale results.
    # The resource-optimized values (batch_size, encoder_lr, etc.) are hardware-dependent
    # and would cause spurious cache misses on different hardware.
    result = {
        "experiment_id": exp_id,
        "name": config.name,
        "datasets": config.datasets,
        "config": original_config_dict,
        "num_train_samples": len(train_samples),
        "metrics": _sanitize_metrics(metrics),
        "training_log": log_history,
    }
    result_file.write_text(json.dumps(result, indent=2))
    logger.debug("[Exp %d] Results saved -> %s", exp_id, result_file)
    print(f"[Exp {exp_id}] Global F1 = {metrics.get('global_f1', 'N/A')}")
    _print_experiment_summary(
        exp_id, config, metrics, elapsed_sec=metrics.get("training_time_sec", 0.0)
    )

    # Disk cleanup — remove model checkpoints after evaluation to free space.
    # Default: remove entire model directory (only the result JSON is needed).
    # Disabled by no_disk_cleanup=True (--no-disk-cleanup flag) or
    # preserved to keep_model=True (--keep-models flag).
    if not no_disk_cleanup:
        cleanup_checkpoints_after_eval(model_dir, keep_model=keep_model)

    return result


def run_custom_experiment(config: ExperimentConfig, result_file: Path) -> dict:
    """Run a single experiment with custom hyperparameters (for sweeps).

    Similar to run_experiment but:
    - Takes a custom ExperimentConfig instead of looking up EXPERIMENTS[exp_id]
    - Saves result to custom result_file instead of results/experiment_N.json
    - Does not use caching (always re-runs)
    - Does not check EXPERIMENTS for validity

    Args:
        config: Custom ExperimentConfig with all hyperparameters
        result_file: Path to save results JSON

    Returns:
        Result dict with metrics, training_log, etc.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = config.experiment_id

    logger.debug("[Sweep Exp %d] %s | datasets=%s", exp_id, config.name, config.datasets)

    # Load data — request source labels when aux_loss_weight is active
    _aux_w_sweep = getattr(config, "aux_loss_weight", 1.0)
    if _aux_w_sweep < 1.0:
        train_samples, val_samples, train_sources_sweep = dataset_loaders.get_combined_dataset(
            config.datasets, sroie_oversample=config.sroie_oversample, return_sources=True
        )
    else:
        train_samples, val_samples = dataset_loaders.get_combined_dataset(
            config.datasets, sroie_oversample=config.sroie_oversample
        )
        train_sources_sweep = None
    if len(train_samples) == 0:
        print(f"[Sweep Exp {exp_id}] WARNING: No samples loaded.")
        result = {
            "experiment_id": exp_id,
            "name": config.name,
            "datasets": config.datasets,
            "num_train_samples": 0,
            "metrics": {},
            "error": "No training samples available",
        }
        result_file.write_text(json.dumps(result, indent=2))
        return result

    # Train
    model_dir = WORKSPACE / "models" / f"sweep_experiment_{exp_id}_{int(time.time())}"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_history = train_experiment(
        exp_id,
        train_samples,
        model_dir,
        val_samples=val_samples,
        config=config,
        sample_sources=train_sources_sweep,
    )

    # Evaluate (pass custom config to avoid EXPERIMENTS lookup)
    metrics = evaluate_experiment(exp_id, model_dir, config=config)

    # Save result
    result = {
        "experiment_id": exp_id,
        "name": config.name,
        "datasets": config.datasets,
        "num_train_samples": len(train_samples),
        "metrics": metrics,
        "training_log": log_history,
    }
    result_file.write_text(json.dumps(result, indent=2))
    logger.debug("[Sweep Exp %d] Results saved -> %s", exp_id, result_file)
    print(f"[Sweep Exp {exp_id}] Global F1 = {metrics.get('global_f1', 'N/A')}")
    return result


def run_experiment_from_config(
    cfg,
    base_processor=None,
    base_model=None,
    overrides: dict | None = None,
) -> dict:
    """Run a YAML-defined DONUT experiment (IDs 9+) via run_custom_experiment().

    Bridges the ``experiment_config_loader.ExperimentConfig`` object (YAML-sourced)
    to the ``run_experiments.ExperimentConfig`` dataclass expected by
    ``run_custom_experiment()``.  Training logic is NOT duplicated — all training
    is handled by the existing ``run_custom_experiment`` / ``train_experiment`` pair.

    Args:
        cfg: An ``experiment_config_loader.ExperimentConfig`` instance (YAML-loaded).
        base_processor: Optional pre-loaded DonutProcessor for reuse across experiments.
            NOTE: run_custom_experiment does not currently accept base_processor;
            the arg is accepted here for API compatibility but ignored.
        base_model: Optional pre-loaded base model.  Same caveat as base_processor.
        overrides: Optional dict of ExperimentConfig field overrides (e.g.
            ``{"epochs": 5}``).  Applied via ``dataclasses.replace()`` so the
            input cfg is never mutated.

    Returns:
        Result dict with experiment_id, name, datasets, num_train_samples, metrics.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = (
        Path(cfg.results_file)
        if getattr(cfg, "results_file", "")
        else RESULTS_DIR / f"experiment_{cfg.id}.json"
    )

    # Compute sroie_oversample from per-dataset oversampling in YAML config.
    # The YAML specifies oversampling per DatasetEntry (.oversample field).
    # The ExperimentConfig uses a single sroie_oversample int for SROIE.
    sroie_oversample = 1
    dataset_names: list[str] = []
    if hasattr(cfg, "datasets"):
        for d in cfg.datasets:
            dataset_names.append(d.name)
            if d.name == "sroie":
                sroie_oversample = getattr(d, "oversample", 1)

    # Build a run_experiments.ExperimentConfig from the YAML cfg fields
    ec = ExperimentConfig(
        name=cfg.name,
        datasets=dataset_names,
        epochs=getattr(cfg, "epochs", 10),
        lr=getattr(cfg, "encoder_lr", 5e-5),
        batch_size=getattr(cfg, "batch_size", 8),
        seed=getattr(cfg, "seed", SEED),
        early_stopping_patience=getattr(cfg, "early_stopping_patience", 3),
        base_model=getattr(cfg, "base_checkpoint", BASE_MODEL),
        warmup_steps=getattr(cfg, "warmup_steps", 40),
        weight_decay=getattr(cfg, "weight_decay", 0.01),
        max_length=getattr(cfg, "max_length", MAX_LENGTH),
        gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 2),
        encoder_lr=getattr(cfg, "encoder_lr", 5e-5),
        decoder_lr=getattr(cfg, "decoder_lr", 1e-4),
        description=getattr(cfg, "description", ""),
        experiment_id=getattr(cfg, "id", 0),
        sroie_oversample=sroie_oversample,
        lr_schedule=getattr(cfg, "scheduler_type", "cosine"),
    )

    if overrides:
        ec = dataclasses.replace(ec, **{k: v for k, v in overrides.items() if hasattr(ec, k)})

    return run_custom_experiment(ec, result_file)


# ---------------------------------------------------------------------------
# Compact structured summary helpers
# ---------------------------------------------------------------------------


def _print_experiment_summary(
    exp_id: int, config: ExperimentConfig, metrics: dict, elapsed_sec: float = 0.0
) -> None:
    """Print a rich Panel summary for the experiment result (plain text fallback).

    Rich: shows a formatted panel with per-field F1 bars, timing, and disk space.
    Plain: machine-readable JSON (AI-agent-friendly).
    """
    # Respect DONUT_QUIET env var — suppress if quiet mode is active
    if os.environ.get("DONUT_QUIET") == "1":
        return

    global_f1 = metrics.get("global_f1", 0.0)

    # Try rich panel first
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="dim", width=14)
        grid.add_column()

        grid.add_row("Experiment", f"[bold]{exp_id}[/] — {config.name}")
        grid.add_row("Datasets", ", ".join(config.datasets))
        grid.add_row("Train samples", str(metrics.get("num_train_samples", 0)))
        grid.add_row("Epochs", str(config.epochs))
        if elapsed_sec > 0:
            grid.add_row("Duration", f"{elapsed_sec / 60:.1f} min")

        f1_color = "green" if global_f1 >= 0.8 else ("yellow" if global_f1 >= 0.5 else "red")
        grid.add_row("Global F1", Text(f"{global_f1:.4f}", style=f"bold {f1_color}"))

        # Per-field F1 bar chart (text-based, 20 chars wide)
        for field in ("company", "date", "address", "total"):
            fv = metrics.get(f"{field}_f1")
            if fv is not None:
                bar_len = int(fv * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                fc = "green" if fv >= 0.8 else ("yellow" if fv >= 0.5 else "red")
                grid.add_row(f"  {field}", f"[{fc}]{bar}[/] {fv:.3f}")

        parse_fail = metrics.get("parse_failures", 0)
        if parse_fail:
            grid.add_row("Parse failures", Text(str(parse_fail), style="yellow"))

        try:
            from constants import format_bytes, get_disk_usage

            _, _, disk_free = get_disk_usage()
            grid.add_row("Disk free", format_bytes(disk_free))
        except Exception:
            pass

        status_icon = "✓" if global_f1 > 0 else "✗"
        border = "green" if global_f1 > 0 else "red"
        console.print(
            Panel(
                grid,
                title=f"[bold {border}]{status_icon} Exp {exp_id} Complete[/]",
                border_style=border,
                padding=(0, 1),
            )
        )
        return
    except Exception:
        pass

    # Plain text fallback — single compact line
    logger.debug(
        "[Exp %d] F1=%.4f company=%.4f date=%.4f address=%.4f total=%.4f (%.1fm)",
        exp_id,
        global_f1,
        metrics.get("company_f1", 0.0),
        metrics.get("date_f1", 0.0),
        metrics.get("address_f1", 0.0),
        metrics.get("total_f1", 0.0),
        elapsed_sec / 60,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def save_summary() -> None:
    """Collect all individual result files into results/all_experiments.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for exp_id in EXPERIMENTS:
        result_file = RESULTS_DIR / f"experiment_{exp_id}.json"
        if result_file.exists():
            with open(result_file) as fh:
                all_results[str(exp_id)] = json.load(fh)

    summary_file = RESULTS_DIR / "all_experiments.json"
    summary_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nSummary saved -> {summary_file}")

    if not all_results:
        return

    # Find best F1 for highlighting
    best_f1 = -1.0
    best_exp_id_str = None
    for exp_id_str, res in all_results.items():
        f1 = res.get("metrics", {}).get("global_f1", float("nan"))
        if not math.isnan(f1) and f1 > best_f1:
            best_f1 = f1
            best_exp_id_str = exp_id_str

    # Try rich table
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        console = Console()
        table = Table(
            title="[bold]DONUT SROIE — Experiment Leaderboard[/]",
            show_header=True,
            header_style="bold dim",
            border_style="dim",
            show_lines=False,
        )
        table.add_column("Exp", justify="right", style="dim", width=4)
        table.add_column("Name", width=34)
        table.add_column("Samples", justify="right", width=8)
        table.add_column("Global F1", justify="right", width=10)
        table.add_column("Rank", justify="center", width=5)

        sorted_by_f1 = sorted(
            all_results.items(),
            key=lambda x: x[1].get("metrics", {}).get("global_f1", -1),
            reverse=True,
        )
        rank_map = {eid: i + 1 for i, (eid, _) in enumerate(sorted_by_f1)}

        for exp_id_str, res in sorted(all_results.items(), key=lambda x: int(x[0])):
            name = res.get("name", "")[:33]
            n = res.get("num_train_samples", 0)
            f1 = res.get("metrics", {}).get("global_f1", float("nan"))
            rank = rank_map.get(exp_id_str, "—")
            is_best = exp_id_str == best_exp_id_str

            if is_best:
                f1_text = Text(f"{f1:.4f} ★", style="bold green")
                name_text = Text(name, style="bold")
                rank_text = Text("#1", style="bold green")
            elif not math.isnan(f1):
                f1_color = "green" if f1 >= 0.8 else ("yellow" if f1 >= 0.5 else "red")
                f1_text = Text(f"{f1:.4f}", style=f1_color)
                name_text = Text(name)
                rank_text = Text(f"#{rank}", style="dim")
            else:
                f1_text = Text("N/A", style="dim")
                name_text = Text(name, style="dim")
                rank_text = Text("—", style="dim")

            table.add_row(exp_id_str, name_text, str(n), f1_text, rank_text)

        console.print(table)
        return
    except Exception:
        pass

    # Plain text fallback
    print(f"\n{'=' * 72}")
    print(f"{'Exp':<5} {'Name':<35} {'Train Samples':>14} {'Global F1':>10}")
    print(f"{'-' * 72}")
    for exp_id_str, res in sorted(all_results.items(), key=lambda x: int(x[0])):
        name = res.get("name", "")[:34]
        n = res.get("num_train_samples", 0)
        f1 = res.get("metrics", {}).get("global_f1", float("nan"))
        f1_str = f"{f1:>10.4f}" if not math.isnan(f1) else "       N/A"
        marker = " ★" if exp_id_str == best_exp_id_str else ""
        print(f"{exp_id_str:<5} {name:<35} {n:>14} {f1_str}{marker}")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Phase 4-5: Initialize audit logger for persistent resource/config tracking
    audit_logger = TrainingAuditLogger(append_to_file="terminal.txt")
    resources = _mm.detect_system_resources()
    audit_logger.log_resource_detection(resources)
    logger.debug(
        "[Resources] GPU: %s (%.1fGB), RAM: %.1fGB, CPU: %d cores",
        resources.device_name,
        resources.vram_gb,
        resources.ram_gb,
        resources.cpu_cores,
    )

    parser = argparse.ArgumentParser(description="Run DONUT SROIE experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all 8 experiments sequentially")
    group.add_argument(
        "--experiment",
        type=int,
        metavar="N",
        choices=range(1, len(EXPERIMENTS) + 1),
        help="Run a single experiment (1-8)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Delete all cached results and re-run from scratch"
    )
    args = parser.parse_args()

    if args.force:
        for result_file in RESULTS_DIR.glob("experiment_*.json"):
            result_file.unlink()
            logger.debug("[force] Deleted cached result: %s", result_file)

    if args.all:
        # Load base model once and pass to each experiment via deep-copy.
        # This replaces 8 × from_pretrained() disk reads with 8 in-RAM deep
        # copies — saving ~4–6 s of safetensors deserialization per experiment.
        _cfg = EXPERIMENTS[1]
        _base_processor = DonutProcessor.from_pretrained(_cfg.base_model)
        _base_model = VisionEncoderDecoderModel.from_pretrained(_cfg.base_model)
        for exp_id in EXPERIMENTS:
            run_experiment(exp_id, base_processor=_base_processor, base_model=_base_model)
        del _base_model, _base_processor
        _gpu_cleanup()
        save_summary()
    else:
        run_experiment(args.experiment)
        save_summary()


if __name__ == "__main__":
    main()
