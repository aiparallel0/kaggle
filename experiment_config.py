# =============================================================================
# experiment_config.py
# Purpose: Merged experiment config module — YAML loader + ablation control suite
# Merged from: experiment_config_loader.py + control_suite.py
# =============================================================================
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
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DatasetEntry",
    "load_experiment",
    "load_all_experiments",
    "load_experiment_selection",
    # ExperimentConfig is intentionally NOT exported here.
    # Callers should use run_experiments.ExperimentConfig for hardcoded experiments.
    # experiment_config_loader.ExperimentConfig is for YAML-loaded experiments only.
]

# ---------------------------------------------------------------------------
# Module-level guard: verify canonical ExperimentConfig consistency
# ---------------------------------------------------------------------------
try:
    from run_experiments import ExperimentConfig as _CanonicalConfig  # noqa: E402

    if not hasattr(_CanonicalConfig, "experiment_id"):
        raise RuntimeError("run_experiments.ExperimentConfig must have 'experiment_id' field")
except ImportError:
    pass  # run_experiments not available in torch-free test envs


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
        depends_on=depends_on,
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


# ---------------------------------------------------------------------------
# Ablation control suite (from control_suite.py)
# ---------------------------------------------------------------------------
from dataclasses import asdict, dataclass, field  # noqa: E402, I001
from typing import ClassVar  # noqa: E402, I001
from constants import BASE_MODEL, MAX_LENGTH, SEED  # noqa: E402, I001

__all__ = [
    "DonutControlConfig",
    "TrOCRControlConfig",
    "YOLOControlConfig",
    "ControlSuite",
    "CONTROL_SUITE",
    "validate_sroie_oversample",
    "get_augmentation_transforms",
]

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
        from control_suite import CONTROL_SUITE

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
