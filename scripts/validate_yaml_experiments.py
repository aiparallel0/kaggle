#!/usr/bin/env python3
"""validate_yaml_experiments.py — Validate all experiment YAML files.

Runs without torch/transformers; uses stdlib + pyyaml only.
Called by the experiments_sroie CI workflow on every pull request.

Usage:
    python scripts/validate_yaml_experiments.py [experiments_dir]

Exit codes:
    0  All files pass validation.
    1  One or more files have errors (details printed to stdout).
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required. Run: pip install pyyaml>=6.0", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Validation rules (mirrors _YAMLExperimentConfig._validate() in run_experiments.py
# but uses only stdlib + pyyaml — no torch dependency)
# ---------------------------------------------------------------------------

VALID_MIXED_PRECISION = {"fp16", "bf16", "fp32"}
VALID_ARCH_TYPES = {"donut", "trocr_yolo"}
VALID_SPLITS = {"train", "all", "val", "test"}
MAX_IMAGE_HEIGHT = 1280  # DONUT native resolution
MAX_IMAGE_WIDTH = 960

# Fields that are sometimes misplaced under model: but belong elsewhere.
# Maps field_name -> correct_section_name.
_MODEL_MISPLACED_FIELDS: dict[str, str] = {
    "image_height": "data",
    "image_width": "data",
    "max_length": "data",
    "max_decode_length": "data",
    "arch_type": "arch",
}


def _validate_file(path: str) -> list[str]:
    """Return a list of error strings for this YAML file (empty = OK)."""
    errors: list[str] = []

    # --- Parse YAML ---
    try:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (yaml.YAMLError, OSError) as exc:
        return [f"Could not read/parse file: {exc}"]

    if not isinstance(raw, dict):
        return ["Top-level value must be a YAML mapping (dict)."]

    # --- Required top-level fields ---
    if "experiment_id" not in raw:
        errors.append("Missing required field: experiment_id")
    if "name" not in raw:
        errors.append("Missing required field: name")

    exp_id = raw.get("experiment_id", "?")
    exp_label = f"Exp {exp_id}"

    # --- model section ---
    model = raw.get("model", {})
    if not isinstance(model, dict):
        errors.append("'model' must be a YAML mapping.")
        model = {}

    # tie_word_embeddings must be false
    twe = model.get("tie_word_embeddings", False)
    if twe:
        errors.append(
            f"{exp_label}: tie_word_embeddings must be false. "
            "Setting true destroys lm_head after resize_token_embeddings(), "
            "causing F1=0.00 on every prediction."
        )

    # Detect fields placed under model: that belong elsewhere
    for misplaced_field, correct_section in _MODEL_MISPLACED_FIELDS.items():
        if misplaced_field in model:
            errors.append(
                f"{exp_label}: '{misplaced_field}' found under model: section "
                f"but belongs under {correct_section}: — it will be silently ignored."
            )

    # --- data section ---
    data = raw.get("data", {})
    if not isinstance(data, dict):
        errors.append("'data' must be a YAML mapping.")
        data = {}

    image_height = data.get("image_height", MAX_IMAGE_HEIGHT)
    image_width = data.get("image_width", MAX_IMAGE_WIDTH)
    allow_high_res = data.get("allow_high_res", False)

    if image_height > MAX_IMAGE_HEIGHT and not allow_high_res:
        errors.append(
            f"{exp_label}: image_height={image_height} exceeds DONUT native {MAX_IMAGE_HEIGHT}. "
            "RAM scales as (H×W)/(1280×960). Set allow_high_res: true to bypass."
        )
    if image_width > MAX_IMAGE_WIDTH and not allow_high_res:
        errors.append(
            f"{exp_label}: image_width={image_width} exceeds DONUT native {MAX_IMAGE_WIDTH}. "
            "RAM scales as (H×W)/(1280×960). Set allow_high_res: true to bypass."
        )

    # --- arch section ---
    arch = raw.get("arch", {})
    if not isinstance(arch, dict):
        errors.append("'arch' must be a YAML mapping.")
        arch = {}

    arch_type = str(arch.get("type", "donut"))
    if arch_type not in VALID_ARCH_TYPES:
        errors.append(
            f"{exp_label}: arch.type='{arch_type}' invalid. "
            f"Choose one of: {sorted(VALID_ARCH_TYPES)}"
        )

    # --- datasets section ---
    tr = raw.get("training", {}) or {}
    is_zero_shot = bool(tr.get("is_zero_shot", False)) or bool(tr.get("skip", False))
    raw_datasets = raw.get("datasets", [])
    if raw_datasets is None:
        raw_datasets = []

    if not isinstance(raw_datasets, list):
        errors.append("'datasets' must be a YAML list.")
    else:
        if not raw_datasets and not is_zero_shot:
            errors.append(
                f"{exp_label}: datasets list is empty. "
                "Set training.is_zero_shot: true if no training data is intended."
            )
        for i, entry in enumerate(raw_datasets):
            if not isinstance(entry, dict):
                errors.append(f"{exp_label}: datasets[{i}] must be a mapping.")
                continue
            if "name" not in entry:
                errors.append(f"{exp_label}: datasets[{i}] missing required 'name' field.")
            split = entry.get("split", "train")
            if split not in VALID_SPLITS:
                errors.append(
                    f"{exp_label}: datasets[{i}].split='{split}' invalid. "
                    f"Choose one of: {sorted(VALID_SPLITS)}"
                )
            oversample = entry.get("oversample", 1)
            if not isinstance(oversample, int) or oversample < 1:
                errors.append(
                    f"{exp_label}: datasets[{i}].oversample={oversample!r} must be an integer ≥ 1."
                )

    # --- training section ---
    if not isinstance(tr, dict):
        errors.append("'training' must be a YAML mapping.")
        tr = {}

    mixed_precision = str(tr.get("mixed_precision", "fp16"))
    if mixed_precision not in VALID_MIXED_PRECISION:
        errors.append(
            f"{exp_label}: training.mixed_precision='{mixed_precision}' invalid. "
            f"Choose one of: {sorted(VALID_MIXED_PRECISION)}"
        )

    # --- validation section ---
    val = raw.get("validation", {}) or {}
    if val.get("precompute_tensors", False):
        errors.append(
            f"{exp_label}: validation.precompute_tensors must be false. "
            "Setting true causes OOM (see memory_manager.py § Val Dataset Rule)."
        )

    return errors


def validate_directory(experiments_dir: str | Path) -> int:
    """Validate all YAML files in experiments_dir. Returns exit code (0=OK, 1=errors)."""
    experiments_dir = Path(experiments_dir)
    pattern_yaml = str(experiments_dir / "*.yaml")
    pattern_yml = str(experiments_dir / "*.yml")
    paths = sorted(glob.glob(pattern_yaml) + glob.glob(pattern_yml))

    if not paths:
        print(f"ERROR: No YAML files found in '{experiments_dir}'.", file=sys.stderr)
        return 1

    all_errors: dict[str, list[str]] = {}

    # --- Per-file validation ---
    for p in paths:
        errs = _validate_file(p)
        if errs:
            all_errors[p] = errs

    # --- Cross-file checks ---
    id_to_files: dict[int, list[str]] = {}
    name_to_files: dict[str, list[str]] = {}

    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError:
            continue  # already reported in per-file pass
        if not isinstance(raw, dict):
            continue

        eid = raw.get("experiment_id")
        if eid is not None:
            id_to_files.setdefault(eid, []).append(p)

        name = raw.get("name")
        if name is not None:
            name_to_files.setdefault(name, []).append(p)

    for eid, files in id_to_files.items():
        if len(files) > 1:
            msg = (
                f"Duplicate experiment_id={eid} across files: "
                + ", ".join(Path(f).name for f in files)
                + ". Each experiment must have a unique integer ID."
            )
            for f in files:
                all_errors.setdefault(f, []).append(msg)

    for name, files in name_to_files.items():
        if len(files) > 1:
            msg = (
                f"Duplicate experiment name '{name}' across files: "
                + ", ".join(Path(f).name for f in files)
                + ". Each experiment must have a unique name."
            )
            for f in files:
                all_errors.setdefault(f, []).append(msg)

    # --- Report ---
    total_files = len(paths)
    failed_files = len(all_errors)

    if not all_errors:
        print(f"OK — {total_files} experiment YAML file(s) passed validation.")
        return 0

    print(f"FAILED — {failed_files}/{total_files} file(s) have errors:\n")
    for path, errs in sorted(all_errors.items()):
        print(f"  {path}:")
        for err in errs:
            print(f"    - {err}")
    print()
    return 1


def main() -> int:
    experiments_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments"
    return validate_directory(experiments_dir)


if __name__ == "__main__":
    sys.exit(main())
