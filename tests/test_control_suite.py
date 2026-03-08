# =============================================================================
# tests/test_control_suite.py
# Purpose: Unit tests for the control_suite parameter hub
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# =============================================================================
"""
Unit tests for control_suite.py.

All tests are designed to run WITHOUT torch, transformers, or GPU dependencies
so they can be collected in any environment (CI, lint-only, CPU-only).
Tests validate:
  - Successful import and singleton construction
  - Default values match project config (ExperimentConfig defaults)
  - Previously-missing critical parameters are now explicitly present
  - Commonly-underdocumented parameters are registered and accessible
  - Inspection helpers work without errors
"""

import pytest

# ── Guard: control_suite only depends on constants.py (stdlib + no torch) ──
# These imports must succeed in any environment that has constants.py.
from control_suite import (
    CONTROL_SUITE,
    ControlSuite,
    DonutControlConfig,
    TrOCRControlConfig,
    YOLOControlConfig,
    get_augmentation_transforms,
    validate_sroie_oversample,
)
from constants import BASE_MODEL, MAX_LENGTH, SEED


# ---------------------------------------------------------------------------
# Basic import and construction
# ---------------------------------------------------------------------------


def test_control_suite_singleton_is_correct_type():
    """CONTROL_SUITE module-level singleton is a ControlSuite instance."""
    assert isinstance(CONTROL_SUITE, ControlSuite)
    assert isinstance(CONTROL_SUITE.donut, DonutControlConfig)
    assert isinstance(CONTROL_SUITE.trocr, TrOCRControlConfig)
    assert isinstance(CONTROL_SUITE.yolo, YOLOControlConfig)


def test_control_suite_default_factory():
    """ControlSuite.default() returns a fresh identical instance."""
    suite = ControlSuite.default()
    assert isinstance(suite, ControlSuite)
    assert suite.donut.base_model == CONTROL_SUITE.donut.base_model
    assert suite.trocr.model_id == CONTROL_SUITE.trocr.model_id
    assert suite.yolo.base_model == CONTROL_SUITE.yolo.base_model


# ---------------------------------------------------------------------------
# DONUT — defaults must match ExperimentConfig
# ---------------------------------------------------------------------------


def test_donut_base_model_matches_constants():
    """DonutControlConfig.base_model matches constants.BASE_MODEL."""
    assert CONTROL_SUITE.donut.base_model == BASE_MODEL


def test_donut_max_length_matches_constants():
    """DonutControlConfig.max_length matches constants.MAX_LENGTH."""
    assert CONTROL_SUITE.donut.max_length == MAX_LENGTH


def test_donut_seed_matches_constants():
    """DonutControlConfig.seed matches constants.SEED."""
    assert CONTROL_SUITE.donut.seed == SEED


def test_donut_defaults_match_experiment_config():
    """Key DonutControlConfig defaults match ExperimentConfig defaults.

    ExperimentConfig is the authoritative training config per CLAUDE.md GP-1.
    DonutControlConfig must mirror these values exactly for consistency.
    """
    d = CONTROL_SUITE.donut
    # From ExperimentConfig in run_experiments.py
    assert d.epochs == 10, "DonutControlConfig.epochs must match ExperimentConfig.epochs=10"
    assert d.encoder_lr == 5e-5, "encoder_lr must be 5e-5 (CLAUDE.md §3)"
    assert d.decoder_lr == 1e-4, "decoder_lr must be 1e-4 (2× encoder LR)"
    assert d.batch_size == 8, "batch_size=8 is optimal per CLAUDE.md §3"
    assert d.gradient_accumulation_steps == 2
    assert d.early_stopping_patience == 3
    assert d.weight_decay == 0.01
    assert d.warmup_steps == 40


def test_donut_tie_word_embeddings_is_false():
    """tie_word_embeddings must be False — changing to True causes F1=0.

    See CLAUDE.md §2: After resize_token_embeddings(), tie_word_embeddings=True
    causes tie_weights() on reload to destroy the learned lm_head → F1=0.
    """
    assert CONTROL_SUITE.donut.tie_word_embeddings is False, (
        "tie_word_embeddings must be False — see CLAUDE.md §2 lm_head weight tying bug"
    )


def test_donut_sort_json_key_is_false():
    """sort_json_key must be False for CORD-based datasets.

    Setting True silently corrupts token ordering in preprocessed datasets.
    See finetuning_params.md commonly_underdocumented[].
    """
    assert CONTROL_SUITE.donut.sort_json_key is False, (
        "sort_json_key must be False for SROIE/CORD datasets — silently corrupts token ordering"
    )


def test_donut_input_size_is_canonical():
    """input_size must be the canonical DONUT finetuning resolution [1280, 960]."""
    assert CONTROL_SUITE.donut.input_size == [1280, 960], (
        "input_size must be [1280, 960] (width×height) — DONUT canonical finetuning resolution"
    )


def test_donut_swin_window_size_is_10():
    """swin_window_size must be 10 (matches donut-base pretrain config).

    Changing this forces full weight re-initialization of all attention layers.
    See finetuning_params.md: marked CRITICAL ⚠️ underdocumented.
    """
    assert CONTROL_SUITE.donut.swin_window_size == 10, (
        "swin_window_size=10 must match donut-base pretrain — changing causes full re-init"
    )


def test_donut_gradient_clip_val_is_one():
    """gradient_clip_val is explicitly documented (was previously implicit)."""
    assert CONTROL_SUITE.donut.gradient_clip_val == 1.0


# ---------------------------------------------------------------------------
# TrOCR — critical parameters and previously-missing ones
# ---------------------------------------------------------------------------


def test_trocr_model_id():
    """TrOCR model ID is the printed-text variant."""
    assert CONTROL_SUITE.trocr.model_id == "microsoft/trocr-base-printed"


def test_trocr_input_size_is_fixed():
    """TrOCR input_size is always 384 — FIXED by architecture (not configurable)."""
    assert CONTROL_SUITE.trocr.input_size == 384, (
        "TrOCR input_size is hardcoded at 384×384 by the BEiT/DeiT encoder architecture"
    )


def test_trocr_patch_size_is_fixed():
    """TrOCR patch_size is always 16 — FIXED by architecture."""
    assert CONTROL_SUITE.trocr.patch_size == 16


def test_trocr_weight_decay_nonzero():
    """TrOCR weight_decay must be non-zero.

    PREVIOUSLY MISSING BUG: AdamW was called without weight_decay argument,
    causing PyTorch to use its default of 0.0. Reference value is 1e-4.
    """
    assert CONTROL_SUITE.trocr.weight_decay > 0, (
        "TrOCR weight_decay must be > 0. Before control_suite, AdamW used "
        "PyTorch default of 0 because weight_decay was not passed."
    )
    assert CONTROL_SUITE.trocr.weight_decay == 1e-4


def test_trocr_gradient_checkpointing_enabled():
    """gradient_checkpointing must be True for TrOCR-base to fit in VRAM."""
    assert CONTROL_SUITE.trocr.gradient_checkpointing is True


def test_trocr_use_cache_disabled():
    """use_cache must be False when gradient_checkpointing is True (incompatible)."""
    assert CONTROL_SUITE.trocr.use_cache is False


def test_trocr_grad_ckpt_vram_threshold_gb_is_24():
    """grad_ckpt_vram_threshold_gb must be 24.0 to mirror the DONUT path.

    Cards with VRAM > 24 GB have enough headroom to skip gradient checkpointing;
    cards at or below 24 GB (e.g. RTX 4090 reporting exactly 24.0 GB) must keep
    it enabled.  The strictly-greater-than comparison is intentional.
    """
    assert CONTROL_SUITE.trocr.grad_ckpt_vram_threshold_gb == 24.0


def test_trocr_grad_ckpt_vram_threshold_registered():
    """grad_ckpt_vram_threshold_gb is registered in the control registry."""
    params = CONTROL_SUITE.to_dict()
    assert "grad_ckpt_vram_threshold_gb" in params.get("trocr", {}), (
        "grad_ckpt_vram_threshold_gb must be a field on TrOCRControlConfig "
        "so that CONTROL_SUITE.to_dict() and print_summary() include it."
    )


def test_trocr_lr_scheduler_is_string():
    """lr_scheduler must be a string (was hardcoded 'linear' before control_suite)."""
    assert isinstance(CONTROL_SUITE.trocr.lr_scheduler, str)
    assert CONTROL_SUITE.trocr.lr_scheduler == "linear"


def test_trocr_warmup_ratio_is_valid():
    """warmup_ratio must be in [0, 1]."""
    assert 0 < CONTROL_SUITE.trocr.warmup_ratio <= 0.5


def test_trocr_warmup_init_lr_is_tiny():
    """warmup_init_lr must be a tiny value (prevents instability at step 0).

    See finetuning_params.md: marked ⚠️ underdocumented — absent from tutorials.
    """
    assert CONTROL_SUITE.trocr.warmup_init_lr < 1e-6, (
        "warmup_init_lr must be tiny (default 1e-8) to prevent instability at warmup step 0"
    )


def test_trocr_patience_documented():
    """patience field exists (even if None — early stopping not yet implemented)."""
    # Attribute must exist; None means not implemented yet
    assert hasattr(CONTROL_SUITE.trocr, "patience")


def test_trocr_augmentation_preset_documented():
    """augmentation_preset field exists (⚠️ underdocumented — DA2 is the official preset)."""
    assert hasattr(CONTROL_SUITE.trocr, "augmentation_preset")


def test_trocr_lora_rank_documented():
    """lora_rank field exists (full finetuning currently, PEFT is optional)."""
    assert hasattr(CONTROL_SUITE.trocr, "lora_rank")


# ---------------------------------------------------------------------------
# YOLO — critical parameters and previously-missing ones
# ---------------------------------------------------------------------------


def test_yolo_freeze_is_documented():
    """freeze field exists — the #1 most impactful underdocumented YOLO param.

    PREVIOUSLY MISSING from train_yolo() call entirely (was using Ultralytics
    default of None). See finetuning_params.md: marked CRITICAL ⚠️.
    """
    assert hasattr(CONTROL_SUITE.yolo, "freeze"), (
        "freeze field must exist in YOLOControlConfig — "
        "it is the most impactful underdocumented finetuning parameter"
    )
    # Currently None (full training) — correct for first pass without domain data
    assert CONTROL_SUITE.yolo.freeze is None


def test_yolo_fliplr_is_zero():
    """fliplr must be 0.0 for text detection (left-right orientation matters)."""
    assert CONTROL_SUITE.yolo.fliplr == 0.0, (
        "fliplr must be 0.0 for receipt text detection — "
        "horizontal flipping corrupts text reading direction"
    )


def test_yolo_mosaic_is_reduced_for_receipts():
    """mosaic is reduced from Ultralytics default 1.0 for the receipt domain."""
    assert 0.0 <= CONTROL_SUITE.yolo.mosaic <= 1.0
    assert CONTROL_SUITE.yolo.mosaic < 1.0, (
        "mosaic should be < 1.0 for receipt domain — "
        "full mosaic (4-image grid) distorts document structure"
    )


def test_yolo_close_mosaic_nonzero():
    """close_mosaic must be > 0 (⚠️ critical for mAP convergence).

    Ultralytics default is 10. Setting to 0 prevents final learning stabilisation.
    See finetuning_params.md: CRITICALLY IMPORTANT for final accuracy.
    """
    assert CONTROL_SUITE.yolo.close_mosaic > 0, (
        "close_mosaic must be > 0 — disabling mosaic for last N epochs is critical for mAP convergence"
    )


def test_yolo_amp_enabled():
    """AMP (mixed precision) is enabled by default for VRAM efficiency."""
    assert CONTROL_SUITE.yolo.amp is True


def test_yolo_weight_decay_documented():
    """weight_decay is explicitly documented (was previously using Ultralytics default)."""
    assert CONTROL_SUITE.yolo.weight_decay == 0.0005


def test_yolo_close_mosaic_documented():
    """close_mosaic is explicitly set (was previously relying on Ultralytics default 10)."""
    assert CONTROL_SUITE.yolo.close_mosaic == 10


def test_yolo_rect_is_false():
    """rect must be False — rect=True silently disables DataLoader shuffle.

    See finetuning_params.md: ⚠️ underdocumented — 'rect shuffle conflict'.
    """
    assert CONTROL_SUITE.yolo.rect is False


def test_yolo_loss_weights_documented():
    """box, cls, dfl loss weights are explicitly documented (were previously missing)."""
    assert CONTROL_SUITE.yolo.box == 7.5
    assert CONTROL_SUITE.yolo.cls == 0.5
    assert CONTROL_SUITE.yolo.dfl == 1.5


# ---------------------------------------------------------------------------
# Cross-model inspection helpers
# ---------------------------------------------------------------------------


def test_to_dict_returns_nested_dict():
    """to_dict() returns a properly nested dict with all three model keys."""
    d = CONTROL_SUITE.to_dict()
    assert set(d.keys()) == {"donut", "trocr", "yolo"}
    assert isinstance(d["donut"], dict)
    assert isinstance(d["trocr"], dict)
    assert isinstance(d["yolo"], dict)
    # Spot-check a few values
    assert d["donut"]["base_model"] == BASE_MODEL
    assert d["trocr"]["model_id"] == "microsoft/trocr-base-printed"
    assert d["yolo"]["freeze"] is None


def test_critical_params_returns_dict():
    """critical_params() returns a dict with at least the known critical params."""
    crits = CONTROL_SUITE.critical_params()
    assert isinstance(crits, dict)
    # Known critical params that must be present
    assert "donut.input_size" in crits
    assert "donut.encoder_lr" in crits
    assert "donut.tie_word_embeddings" in crits
    assert "trocr.model_id" in crits
    assert "trocr.learning_rate" in crits
    assert "yolo.freeze" in crits
    assert "yolo.imgsz" in crits
    assert "yolo.mosaic" in crits


def test_underdocumented_params_returns_list():
    """underdocumented_params() returns a non-empty list of parameter names."""
    underdoc = CONTROL_SUITE.underdocumented_params()
    assert isinstance(underdoc, list)
    assert len(underdoc) > 0
    # Known underdocumented params from finetuning_params.md commonly_underdocumented[]
    assert "donut.swin_window_size" in underdoc
    assert "donut.sort_json_key" in underdoc
    assert "donut.align_long_axis" in underdoc
    assert "trocr.warmup_init_lr" in underdoc
    assert "trocr.patience" in underdoc
    assert "trocr.augmentation_preset" in underdoc
    assert "yolo.freeze" in underdoc
    assert "yolo.close_mosaic" in underdoc
    assert "yolo.rect" in underdoc


def test_print_summary_runs_without_error(capsys):
    """print_summary() runs without raising and produces output."""
    CONTROL_SUITE.print_summary()
    captured = capsys.readouterr()
    assert "CONTROL SUITE" in captured.out
    assert "DONUT" in captured.out
    assert "TrOCR" in captured.out
    assert "YOLOv8" in captured.out
    assert "CRITICAL" in captured.out
    assert "freeze" in captured.out


def test_total_param_count_is_comprehensive():
    """Total parameter count covers all three models with significant coverage."""
    d = CONTROL_SUITE.to_dict()
    total = sum(len(v) for v in d.values())
    # At least 60 params total (DONUT ~20, TrOCR ~20, YOLO ~40+)
    assert total >= 60, f"Expected ≥60 documented parameters, found {total}"


# ---------------------------------------------------------------------------
# Root Cause 1 — validate_sroie_oversample
# ---------------------------------------------------------------------------


def test_validate_sroie_oversample_single_dataset_passes():
    """Single-dataset runs are always valid regardless of sroie_oversample."""
    validate_sroie_oversample(["sroie"], sroie_oversample=1)
    validate_sroie_oversample(["sroie"], sroie_oversample=2)
    validate_sroie_oversample(["wildreceipt"], sroie_oversample=1)


def test_validate_sroie_oversample_multi_dataset_sufficient_passes():
    """Multi-dataset runs with sroie_oversample >= 2 must not raise."""
    validate_sroie_oversample(["sroie", "wildreceipt"], sroie_oversample=2)
    validate_sroie_oversample(["sroie", "wildreceipt"], sroie_oversample=3)
    validate_sroie_oversample(["sroie", "wildreceipt", "invoices_donut"], sroie_oversample=2)


def test_validate_sroie_oversample_multi_dataset_insufficient_raises():
    """Multi-dataset runs with sroie_oversample < 2 must raise ValueError."""
    with pytest.raises(ValueError, match="sroie_oversample=1"):
        validate_sroie_oversample(["sroie", "wildreceipt"], sroie_oversample=1)


def test_validate_sroie_oversample_error_message_is_informative():
    """ValueError message must mention both datasets and the required minimum."""
    with pytest.raises(ValueError) as exc_info:
        validate_sroie_oversample(["sroie", "invoices_donut"], sroie_oversample=1)
    msg = str(exc_info.value)
    assert "sroie_oversample" in msg
    assert ">= 2" in msg


def test_validate_sroie_oversample_exported():
    """validate_sroie_oversample must be in control_suite.__all__."""
    import control_suite

    assert "validate_sroie_oversample" in control_suite.__all__


def test_validate_sroie_oversample_skip_guard_waives_check():
    """skip_guard=True must allow multi-dataset + oversample=1 without raising."""
    validate_sroie_oversample(["sroie", "wildreceipt"], sroie_oversample=1, skip_guard=True)
    validate_sroie_oversample(
        ["sroie", "wildreceipt", "invoices_donut"], sroie_oversample=1, skip_guard=True
    )


def test_validate_sroie_oversample_skip_guard_false_still_raises():
    """Explicitly passing skip_guard=False must still raise for bad configs."""
    with pytest.raises(ValueError):
        validate_sroie_oversample(["sroie", "wildreceipt"], sroie_oversample=1, skip_guard=False)


# ---------------------------------------------------------------------------
# Root Cause 2 — TrOCRControlConfig.effective_batch_size
# ---------------------------------------------------------------------------


def test_effective_batch_size_high_vram_returns_max():
    """With ample VRAM (>= reserved + batch * per_item), return full batch_size."""
    cfg = TrOCRControlConfig(batch_size=16)
    # 6.0 reserved + 16 * 0.3 = 6.0 + 4.8 = 10.8 GiB needed; 24 GiB available → no scaling
    result = cfg.effective_batch_size(vram_gb=24.0)
    assert result == 16


def test_effective_batch_size_low_vram_reduces_batch():
    """With constrained VRAM, effective_batch_size returns a value < batch_size."""
    cfg = TrOCRControlConfig(batch_size=16)
    # 8 GiB free: usable = 8 - 6 = 2 GiB, safe = int(2 / 0.3) = 6
    result = cfg.effective_batch_size(vram_gb=8.0)
    assert result < 16
    assert result >= 1


def test_effective_batch_size_minimum_is_one():
    """effective_batch_size never returns less than 1, even for batch_size=1."""
    cfg = TrOCRControlConfig(batch_size=1)
    # min(batch_size=1, safe) == 1 regardless of VRAM
    result = cfg.effective_batch_size(vram_gb=24.0)
    assert result == 1
    result_low = cfg.effective_batch_size(vram_gb=0.1)
    assert result_low >= 1


def test_effective_batch_size_matches_inline_formula():
    """effective_batch_size must produce the same result as the inline formula in train_trocr."""
    cfg = TrOCRControlConfig(batch_size=16)
    free_gb = 10.0
    # Replicate the formula from train_trocr_yolo.py
    reserved = cfg._TROCR_RESERVED_GB
    per_item = cfg._TROCR_PER_ITEM_GB
    usable = max(free_gb - reserved, 1.0)
    expected = min(16, max(1, int(usable / per_item)))
    assert cfg.effective_batch_size(free_gb) == expected


def test_trocr_control_config_has_calibration_constants():
    """TrOCRControlConfig must expose _TROCR_RESERVED_GB and _TROCR_PER_ITEM_GB."""
    assert hasattr(TrOCRControlConfig, "_TROCR_RESERVED_GB")
    assert hasattr(TrOCRControlConfig, "_TROCR_PER_ITEM_GB")
    assert TrOCRControlConfig._TROCR_RESERVED_GB == 6.0
    assert TrOCRControlConfig._TROCR_PER_ITEM_GB == 0.3


# ---------------------------------------------------------------------------
# Root Cause 5 — YOLOControlConfig.recommended_freeze
# ---------------------------------------------------------------------------


def test_recommended_freeze_large_dataset_no_freeze():
    """>= 2000 samples: no freezing (full training)."""
    cfg = YOLOControlConfig()
    assert cfg.recommended_freeze(2000) is None
    assert cfg.recommended_freeze(3940) is None


def test_recommended_freeze_medium_dataset_freeze_backbone():
    """500–1999 samples: freeze backbone (value 10)."""
    cfg = YOLOControlConfig()
    assert cfg.recommended_freeze(500) == 10
    assert cfg.recommended_freeze(1000) == 10
    assert cfg.recommended_freeze(1999) == 10


def test_recommended_freeze_small_dataset_freeze_early_layers():
    """< 500 samples: freeze first 3 backbone layers."""
    cfg = YOLOControlConfig()
    assert cfg.recommended_freeze(499) == 3
    assert cfg.recommended_freeze(100) == 3
    assert cfg.recommended_freeze(0) == 3


def test_recommended_freeze_boundary_exactly_500():
    """Boundary: exactly 500 samples maps to freeze=10."""
    cfg = YOLOControlConfig()
    assert cfg.recommended_freeze(500) == 10


def test_recommended_freeze_boundary_exactly_2000():
    """Boundary: exactly 2000 samples maps to no freeze."""
    cfg = YOLOControlConfig()
    assert cfg.recommended_freeze(2000) is None


# ---------------------------------------------------------------------------
# Root Cause 4 — AST regression: weight_decay wired from CONTROL_SUITE
# ---------------------------------------------------------------------------


def test_train_trocr_yolo_reads_weight_decay_from_control_suite():
    """train_trocr_yolo.py must read weight_decay from CONTROL_SUITE.trocr, not hardcode 0.

    Uses AST inspection to verify the AdamW call passes weight_decay via
    CONTROL_SUITE.trocr (or a local alias thereof), not as a literal 0 or omitted.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).parent.parent / "train_trocr_yolo.py").read_text()
    tree = ast.parse(source)

    # Collect all keyword arguments to AdamW calls
    adamw_weight_decay_values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_adamw = (
                (isinstance(func, ast.Name) and func.id == "AdamW")
                or (isinstance(func, ast.Attribute) and func.attr == "AdamW")
            )
            if is_adamw:
                for kw in node.keywords:
                    if kw.arg == "weight_decay":
                        adamw_weight_decay_values.append(kw.value)

    assert adamw_weight_decay_values, (
        "No AdamW calls with weight_decay= found in train_trocr_yolo.py"
    )

    # At least one AdamW call must read weight_decay from a non-literal source.
    # Acceptable: CONTROL_SUITE.trocr.weight_decay OR a local alias (_trocr.weight_decay
    # where _trocr is assigned CONTROL_SUITE.trocr earlier in the function).
    # Unacceptable: weight_decay=0 (literal 0) or weight_decay omitted entirely.
    def _is_attribute_access(node: ast.expr) -> bool:
        """Return True if node is an attribute access (x.y or x.y.z), not a literal."""
        return isinstance(node, ast.Attribute)

    has_attribute_ref = any(_is_attribute_access(v) for v in adamw_weight_decay_values)
    assert has_attribute_ref, (
        "train_trocr_yolo.py AdamW optimizer must read weight_decay via an attribute "
        "access (e.g. CONTROL_SUITE.trocr.weight_decay or _trocr.weight_decay), not as "
        f"a literal. Found AST nodes: {[ast.dump(v) for v in adamw_weight_decay_values]}"
    )

    # Also verify none of the AdamW weight_decay arguments are literal 0.
    for val in adamw_weight_decay_values:
        if isinstance(val, ast.Constant):
            assert val.value != 0, (
                "train_trocr_yolo.py AdamW weight_decay must not be literal 0 — "
                "read from CONTROL_SUITE.trocr.weight_decay instead."
            )


# ---------------------------------------------------------------------------
# Root Cause 3 — get_augmentation_transforms helper
# ---------------------------------------------------------------------------


def test_get_augmentation_transforms_none_returns_none():
    """get_augmentation_transforms(None) must return None."""
    assert get_augmentation_transforms(None) is None


def test_get_augmentation_transforms_invalid_raises():
    """get_augmentation_transforms with unknown preset must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown augmentation preset"):
        get_augmentation_transforms("DA99")


def test_get_augmentation_transforms_exported():
    """get_augmentation_transforms must be in control_suite.__all__."""
    import control_suite

    assert "get_augmentation_transforms" in control_suite.__all__
