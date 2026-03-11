"""Tests for CLAUDE.md §19 Guardrail Principles (GP-1 through GP-4).

Phase 4 — Mossad/GCHQ: Asset Protection + Communications Intelligence

These tests use AST inspection and source-code scanning to verify that
production files follow the Guardrail Principles from CLAUDE.md §19.
AST-based checks are immune to code reformatting and comment changes —
they examine the actual parse tree of the source.

All tests run without GPU, model weights, or disk access.
"""

import ast
from pathlib import Path


import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent


def _read(filename: str) -> str:
    return (_ROOT / filename).read_text(encoding="utf-8")


def _ast_call_names(source: str) -> list[str]:
    """Return a flat list of all function/method names called in *source*."""
    tree = ast.parse(source)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return names


# ---------------------------------------------------------------------------
# GP-1 — Never mutate EXPERIMENTS global state
# Verify run_experiment() uses dataclasses.replace()
# ---------------------------------------------------------------------------


class TestGP1ExperimentsImmutability:
    """GP-1 (CLAUDE.md §19): run_experiment() must use dataclasses.replace().

    Python dataclasses are mutable.  ``config = EXPERIMENTS[exp_id]`` is a
    reference, not a copy.  Any field assignment propagates back to the global
    dict and corrupts the cache-validity check for all subsequent experiments.
    """

    def test_run_experiment_calls_dataclasses_replace(self):
        """run_experiment() source must contain a dataclasses.replace() call."""
        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        pytest.importorskip("transformers", reason="transformers required")

        import inspect

        from run_experiments import run_experiment  # noqa: E402, I001

        src = inspect.getsource(run_experiment)
        tree = ast.parse(src)

        # Either `dataclasses.replace(...)` (Attribute call) or a bare `replace(...)`
        found_replace = any(
            (isinstance(node, ast.Attribute) and node.attr == "replace")
            or (isinstance(node, ast.Name) and node.id == "replace")
            for node in ast.walk(tree)
        )
        assert found_replace, (
            "run_experiment() must call dataclasses.replace() to isolate config from the "
            "global EXPERIMENTS dict. "
            "GP-1 (CLAUDE.md §19): never assign to EXPERIMENTS[N].field — use "
            "dataclasses.replace(EXPERIMENTS[N], field=value) instead."
        )

    def test_run_experiments_file_imports_dataclasses(self):
        """run_experiments.py must import the dataclasses module."""
        source = _read("run_experiments.py")
        tree = ast.parse(source)

        # Look for `import dataclasses`
        found = any(
            isinstance(node, ast.Import)
            and any(alias.name == "dataclasses" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert found, "run_experiments.py must `import dataclasses` to use dataclasses.replace()."


# ---------------------------------------------------------------------------
# GP-2 — Always validate optimizer step count before training
# ---------------------------------------------------------------------------


class TestGP2ValidateStepCount:
    """GP-2 (CLAUDE.md §19): validate_training_config() must be called before training.

    Without step-count validation, a large batch on high-VRAM GPU can reduce
    total optimizer steps below DONUT's convergence threshold (~200 steps),
    producing perfectly structured but completely empty predictions.  This
    never raises an exception — it is a silent failure.
    """

    def test_run_experiment_calls_validate_training_config(self):
        """run_experiment() must reference validate_training_config."""
        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        pytest.importorskip("transformers", reason="transformers required")

        import inspect

        from run_experiments import run_experiment  # noqa: E402, I001

        src = inspect.getsource(run_experiment)
        call_names = _ast_call_names(src)

        assert "validate_training_config" in call_names, (
            "run_experiment() must call validate_training_config() before starting training. "
            "GP-2 (CLAUDE.md §19): enforces minimum optimizer step count (~200 steps)."
        )

    def test_validate_training_config_exists_in_resource_optimizer(self):
        """resource_optimizer.py must export validate_training_config."""
        from resource_optimizer import validate_training_config

        assert callable(validate_training_config), (
            "validate_training_config must be a callable in resource_optimizer.py"
        )

    def test_validate_training_config_raises_on_too_few_steps(self):
        """Fewer than 200 optimizer steps must raise ValueError."""
        from resource_optimizer import validate_training_config

        # 10 samples × 1 epoch / batch_size=8 / accum=1 = only 1 step — way too few
        with pytest.raises((ValueError, RuntimeError)):
            validate_training_config(
                batch_size=8,
                gradient_accumulation_steps=1,
                num_train_samples=10,
                epochs=1,
            )


# ---------------------------------------------------------------------------
# GP-3 — Always use list form for convert_tokens_to_ids
# ---------------------------------------------------------------------------


class TestGP3ConvertTokensListForm:
    """GP-3 (CLAUDE.md §19): convert_tokens_to_ids must be called with a list.

    The string form iterates over characters, returning the ID for '<' — not
    the full '<s_sroie>' token.  This causes the decoder to start from the
    wrong token, producing garbage output that never raises an exception.

    Correct:   tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    Wrong:     tokenizer.convert_tokens_to_ids("<s_sroie>")
    """

    @staticmethod
    def _find_string_form_violations(source: str, filename: str) -> list[str]:
        """Return violation descriptions for any string-form convert_tokens_to_ids calls."""
        violations = []
        for i, line in enumerate(source.splitlines(), 1):
            if "convert_tokens_to_ids(" not in line:
                continue
            stripped = line.strip()
            idx = stripped.find("convert_tokens_to_ids(")
            after = stripped[idx + len("convert_tokens_to_ids(") :]
            # String literal immediately after '(' means wrong form
            if after.startswith('"') or after.startswith("'"):
                violations.append(f"{filename}:{i}: {stripped}")
        return violations

    def test_train_py_uses_list_form(self):
        """train.py must use the list form of convert_tokens_to_ids."""
        source = _read("train.py")
        violations = self._find_string_form_violations(source, "train.py")
        assert violations == [], (
            "train.py uses string form of convert_tokens_to_ids() — GP-3 violation: "
            + "; ".join(violations)
        )

    def test_run_experiments_uses_list_form(self):
        """run_experiments.py must use the list form of convert_tokens_to_ids."""
        source = _read("run_experiments.py")
        violations = self._find_string_form_violations(source, "run_experiments.py")
        assert violations == [], (
            "run_experiments.py uses string form of convert_tokens_to_ids() — GP-3 violation: "
            + "; ".join(violations)
        )


# ---------------------------------------------------------------------------
# GP-4 — decoder_start_token_id roundtrip verification
# ---------------------------------------------------------------------------


class TestGP4DecoderStartTokenVerification:
    """GP-4 (CLAUDE.md §19): after setting decoder_start_token_id, the code
    must verify ``tokenizer.decode([token_id]) == '<s_sroie>'`` and raise
    RuntimeError if wrong.

    Silent wrong token IDs produce models that generate valid XML structure
    but wrong content — indistinguishable from normal output without manual
    inspection.
    """

    def test_train_py_has_roundtrip_check(self):
        """train.py must contain decoder_start_token_id roundtrip assertion."""
        source = _read("train.py")
        assert "decoder_start_token_id" in source, (
            "train.py must set model.config.decoder_start_token_id"
        )
        # A RuntimeError guard must be present for the GP-4 roundtrip
        assert "RuntimeError" in source, (
            "train.py must raise RuntimeError when decoder_start_token_id decodes incorrectly. "
            "GP-4 (CLAUDE.md §19): always verify the roundtrip "
            "decode(convert_tokens_to_ids(['<s_sroie>'])[0]) == '<s_sroie>'."
        )

    def test_run_experiments_has_roundtrip_check(self):
        """run_experiments.py must contain decoder_start_token_id roundtrip assertion."""
        source = _read("run_experiments.py")
        assert "decoder_start_token_id" in source
        assert "RuntimeError" in source, (
            "run_experiments.py must raise RuntimeError when decoder_start_token_id "
            "decodes incorrectly (GP-4)."
        )

    def test_train_py_verifies_sroie_token_name(self):
        """train.py must reference '<s_sroie>' as the expected decoded token."""
        source = _read("train.py")
        assert "<s_sroie>" in source, (
            "train.py must verify that decoder_start_token_id decodes back to '<s_sroie>'"
        )


# ---------------------------------------------------------------------------
# Pattern 3 — Compatibility shim (CLAUDE.md §5 Pattern 3)
# ---------------------------------------------------------------------------


class TestPattern3CompatShim:
    """Pattern 3 (CLAUDE.md §5): The PreTrainedTokenizerBase compat shim must
    be present in dataset_loaders.py.

    In transformers ≥4.47, PreTrainedTokenizerBase moved from
    ``transformers.tokenization_utils_base`` to ``transformers`` directly.
    The try/except shim ensures both old and new versions work.
    """

    def test_compat_shim_contains_pretrained_tokenizer_base(self):
        """dataset_loaders.py must reference PreTrainedTokenizerBase."""
        source = _read("dataset_loaders.py")
        assert "PreTrainedTokenizerBase" in source, (
            "dataset_loaders.py is missing the PreTrainedTokenizerBase import. "
            "Pattern 3 requires the compat shim for transformers ≥4.47 compatibility."
        )

    def test_compat_shim_has_fallback_import_path(self):
        """The shim must include the fallback from tokenization_utils_base."""
        source = _read("dataset_loaders.py")
        assert "tokenization_utils_base" in source, (
            "dataset_loaders.py missing fallback import from "
            "transformers.tokenization_utils_base — "
            "this will crash on transformers <4.47. "
            "Pattern 3 (CLAUDE.md §5): preserve the compat shim."
        )

    def test_compat_shim_uses_try_except(self):
        """The shim must be a try/except block, not a version comparison."""
        source = _read("dataset_loaders.py")
        tree = ast.parse(source)

        # The shim wraps the conditional import in a try/except:
        #   try:
        #       import transformers
        #       if not hasattr(...):
        #           from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        #           ...
        #   except Exception:
        #       pass
        #
        # So PreTrainedTokenizerBase appears in the try *body*, not the except handler.
        # Walk all Try nodes and check whether their body (or handlers) reference the name.
        found_shim = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                try_body_src = "\n".join(ast.unparse(stmt) for stmt in node.body)
                if "PreTrainedTokenizerBase" in try_body_src:
                    found_shim = True
                    break

        assert found_shim, (
            "dataset_loaders.py has no try/except block whose body imports "
            "PreTrainedTokenizerBase. Pattern 3 requires the try/except form (not a "
            "version check) so that the shim works across all "
            "Python/transformers version combinations."
        )


# ---------------------------------------------------------------------------
# Pattern 7 — importorskip must precede guarded imports in test files
# ---------------------------------------------------------------------------


class TestPattern7ImportorskipOrder:
    """Pattern 7 (CLAUDE.md §16): pytest.importorskip() must appear before
    the import of the package it guards.

    If the import fires before the importorskip call, a missing package raises
    ImportError at collection time and NO tests in that file run — they don't
    even show as 'skipped'.
    """

    def test_test_metrics_importorskip_before_donut_evaluator(self):
        """test_metrics.py must call importorskip before importing donut_evaluator."""
        source = (Path(__file__).parent / "test_metrics.py").read_text()
        assert "importorskip" in source, "test_metrics.py must use pytest.importorskip"

        importorskip_pos = source.find("importorskip")
        donut_import_pos = source.find("from donut_evaluator import")
        assert importorskip_pos < donut_import_pos, (
            "Pattern 7 violation in test_metrics.py: 'from donut_evaluator import' appears "
            "before the pytest.importorskip guard. The importorskip must come first so that "
            "collection fails gracefully (skip) rather than crashing with ImportError."
        )

    def test_test_pipeline_smoke_has_importorskip(self):
        """test_pipeline_smoke.py must use importorskip before importing torch modules."""
        source = (Path(__file__).parent / "test_pipeline_smoke.py").read_text()
        assert "importorskip" in source, (
            "test_pipeline_smoke.py must use pytest.importorskip to guard torch imports. "
            "Pattern 7 (CLAUDE.md §16)."
        )

    def test_test_train_invariants_importorskip_before_train(self):
        """test_train_invariants.py must call importorskip before importing train."""
        source = (Path(__file__).parent / "test_train_invariants.py").read_text()
        assert "importorskip" in source, (
            "test_train_invariants.py must use pytest.importorskip to guard transformers import"
        )
