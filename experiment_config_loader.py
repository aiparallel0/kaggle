"""
experiment_config_loader.py
============================
Loads per-experiment YAML files from the experiments/ directory and converts
them into ExperimentConfig dataclasses for use by run_experiments.py and
run_all.py.

Usage
-----
    from experiment_config_loader import load_experiment, load_all_experiments

    # Load a single experiment by ID:
    cfg = load_experiment(6)

    # Load all experiments from the directory:
    configs = load_all_experiments("experiments/")

    # Load specific experiments:
    configs = load_all_experiments("experiments/", experiment_ids=[1, 6])

    # Load only enabled experiments from experiment_selection.json:
    configs = load_experiment_selection()

Design
------
Each YAML file is self-describing and fully autonomous.  There is no
inheritance, no shared base config, and no required ordering.  Adding a new
experiment means dropping a new YAML file in the experiments/ directory —
no other file needs to be touched.

The loader performs strict validation so that malformed configs are caught
before any training is launched, not silently ignored mid-run.

The ExperimentConfig dataclass produced here is compatible with DonutTrainer
and the run_experiments.py orchestrator via duck-typed attribute access.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML is required: pip install pyyaml") from e


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
# ExperimentConfig — the authoritative config object consumed by training code
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ExperimentConfig:
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
    warmup_steps: int = 500

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
            warnings.warn(
                f"\n{'=' * 60}\n"
                f"WARNING: Exp {self.id} has image_height={self.image_height} "
                f"(>{1280}). allow_high_res=True bypasses the guard.\n"
                f"RAM scales as (H*W)/(1280*960). "
                f"At 2560x1920 this is 4x.\n"
                f"Ensure processor_config.json is updated BEFORE training.\n"
                f"REVERT processor_config.json AFTER this experiment.\n"
                f"{'=' * 60}",
                stacklevel=3,
            )
        if self.image_width > 960 and self.allow_high_res:
            warnings.warn(
                f"\n{'=' * 60}\n"
                f"WARNING: Exp {self.id} has image_width={self.image_width} "
                f"(>{960}). allow_high_res=True bypasses the guard.\n"
                f"RAM scales as (H*W)/(1280*960). "
                f"At 2560x1920 this is 4x.\n"
                f"Ensure processor_config.json is updated BEFORE training.\n"
                f"REVERT processor_config.json AFTER this experiment.\n"
                f"{'=' * 60}",
                stacklevel=3,
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
    with open(path, "r", encoding="utf-8") as fh:
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
) -> ExperimentConfig:
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
) -> list[ExperimentConfig]:
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

    configs: list[ExperimentConfig] = []
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


def _yaml_to_config(path: str | Path) -> ExperimentConfig:
    """Parse one YAML file into an ExperimentConfig."""
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
    warmup_steps = int(sched.get("warmup_steps", 500))

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

    return ExperimentConfig(
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
    )


def load_experiment_selection(
    selection_file: str | Path = "experiment_selection.json",
    experiments_dir: str | Path = "experiments",
) -> list[ExperimentConfig]:
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

    with open(selection_file, "r", encoding="utf-8") as fh:
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
# Quick self-test — run directly: python experiment_config_loader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    experiments_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments")

    print(f"Loading all experiments from: {experiments_dir.resolve()}")
    try:
        configs = load_all_experiments(experiments_dir)
    except Exception as exc:
        print(f"\n✗ FAILED: {exc}")
        sys.exit(1)

    print(f"\nLoaded {len(configs)} experiment config(s):\n")
    for cfg in configs:
        ds_summary = (
            ", ".join(
                f"{d.name}" + (f"×{d.oversample}" if d.oversample > 1 else "") for d in cfg.datasets
            )
            or "(zero-shot — no datasets)"
        )
        print(
            f"  Exp {cfg.id:2d}  {cfg.name:<40s}  "
            f"arch={cfg.arch_type:<12s}  "
            f"zero_shot={cfg.is_zero_shot!s:<5s}  "
            f"datasets=[{ds_summary}]  "
            f"bs={cfg.batch_size}×{cfg.gradient_accumulation_steps}  "
            f"epochs={cfg.epochs}  "
            f"eff_batch={cfg.effective_batch_size}"
        )

    print("\n✓ All configs loaded and validated successfully.")

    # Test load_experiment_selection()
    sel_file = Path("experiment_selection.json")
    if sel_file.exists():
        print(f"\nTesting load_experiment_selection() from {sel_file}...")
        try:
            sel_configs = load_experiment_selection(sel_file, experiments_dir)
            print(f"  Selected {len(sel_configs)} experiment(s): {[c.id for c in sel_configs]}")
            print("✓ load_experiment_selection() passed.")
        except Exception as exc:
            print(f"✗ load_experiment_selection() FAILED: {exc}")
            sys.exit(1)
    else:
        print(f"\n  (Skipping load_experiment_selection() — {sel_file} not found)")
