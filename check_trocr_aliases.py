#!/usr/bin/env python3
"""check_trocr_aliases.py — Standalone diagnostic for microsoft/trocr-base-printed.

Run with:
    python check_trocr_aliases.py [--hf-token TOKEN]

Checks:
  1. Environment info (Python / torch / transformers / safetensors versions)
  2. Load microsoft/trocr-base-printed on CPU
  3. Scan state_dict for shared-memory (tied-weight) aliases
  4. Test broken path: safetensors.save_file on raw state dict
  5. Test fixed path: generic clone-loop then safetensors.save_file
  6. Verify decoder.lm_head.weight is present in the saved file
  7. PASS / FAIL summary

No GPU required; all operations run on CPU.
No dependencies beyond requirements.txt (torch, transformers, safetensors).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1 — Environment info
# ---------------------------------------------------------------------------

print("=" * 70)
print("check_trocr_aliases.py — TrOCR safetensors alias diagnostic")
print("=" * 70)
print()
print(f"Python  : {sys.version}")

try:
    import torch

    print(f"torch   : {torch.__version__}")
except ImportError:
    print("torch   : NOT INSTALLED — install with: pip install torch")
    sys.exit(1)

try:
    import transformers

    print(f"transformers : {transformers.__version__}")
except ImportError:
    print("transformers : NOT INSTALLED — install with: pip install transformers")
    sys.exit(1)

try:
    import safetensors

    print(f"safetensors  : {safetensors.__version__}")
    from safetensors import safe_open as _sf_safe_open  # noqa: PLC0415
    from safetensors.torch import save_file as _st_save_file  # noqa: PLC0415
except ImportError:
    print("safetensors  : NOT INSTALLED — install with: pip install safetensors")
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(
    description="Diagnostic: check TrOCR safetensors alias behaviour",
    add_help=True,
)
_parser.add_argument(
    "--hf-token",
    default=None,
    metavar="TOKEN",
    help="HuggingFace token for gated model access (optional)",
)
_args = _parser.parse_args()

# If a token was given on the command line, pass it; otherwise fall back to
# HF_TOKEN env var or an hf_token.txt file in the current directory.
_hf_token: str | None = _args.hf_token
if _hf_token is None:
    _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if _hf_token is None:
    _token_file = Path("hf_token.txt")
    if _token_file.exists():
        _hf_token = _token_file.read_text().strip() or None

_MODEL_ID = "microsoft/trocr-base-printed"

# ---------------------------------------------------------------------------
# Step 2 — Load the model
# ---------------------------------------------------------------------------

print(f"[Step 2] Loading {_MODEL_ID} on CPU …")
print("         (this may take 30–60 s on a slow connection)")
print()

try:
    from transformers import VisionEncoderDecoderModel  # noqa: PLC0415

    _load_kwargs: dict = {
        "low_cpu_mem_usage": False,
        "torch_dtype": torch.float32,
    }
    if _hf_token:
        _load_kwargs["token"] = _hf_token

    model = VisionEncoderDecoderModel.from_pretrained(_MODEL_ID, **_load_kwargs)
    model.eval()
    print(f"  Loaded {_MODEL_ID} successfully.")
    print()
except OSError as exc:
    print(f"  ERROR (network / file-system): {exc}")
    print()
    print("  Possible causes:")
    print("   • No internet access on this machine")
    print("   • Model requires a HuggingFace token: pass --hf-token <TOKEN>")
    sys.exit(1)
except RuntimeError as exc:
    print(f"  ERROR loading model: {exc}")
    print()
    print("  Possible causes:")
    print("   • transformers version too old for this model")
    print("   • Model requires a HuggingFace token: pass --hf-token <TOKEN>")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 3 — Scan for ALL aliases
# ---------------------------------------------------------------------------

print("[Step 3] Scanning state_dict for shared-memory aliases …")
print()

sd_original = model.state_dict()
_seen: dict[int, str] = {}
_aliases: list[tuple[str, str]] = []  # (duplicate_key, first_key)

for key, tensor in sd_original.items():
    ptr = tensor.data_ptr()
    if ptr in _seen:
        _aliases.append((key, _seen[ptr]))
    else:
        _seen[ptr] = key

if _aliases:
    for dup_key, first_key in _aliases:
        print(f"  ALIAS: {dup_key}  shares storage with  {first_key}")
    print()
    print(f"  Total aliases found: {len(_aliases)}")
else:
    print("  OK: No shared-memory aliases found in state dict.")
print()

# ---------------------------------------------------------------------------
# Step 4 — Test the BROKEN path
# ---------------------------------------------------------------------------

print("[Step 4] Testing BROKEN path (raw state dict → safetensors.save_file) …")
print()

# Shallow copy: intentionally keeps the same tensor references so that
# shared-memory aliases are preserved — this is what we want to test.
sd_broken = {k: v for k, v in sd_original.items()}

_broken_tmp = Path(tempfile.gettempdir()) / "test_broken.safetensors"
_broken_succeeded = False

try:
    _st_save_file(sd_broken, str(_broken_tmp))
    _broken_succeeded = True
    print("  BROKEN PATH: No aliases — save_file succeeded even without cloning.")
except RuntimeError as exc:
    print("  BROKEN PATH: safetensors.save_file with raw state dict → FAILS as expected")
    print(f"  RuntimeError: {exc}")
print()

# ---------------------------------------------------------------------------
# Step 5 — Test the FIXED path (generic clone loop)
# ---------------------------------------------------------------------------

print("[Step 5] Testing FIXED path (generic clone loop → safetensors.save_file) …")
print()

sd_fixed = {k: v for k, v in sd_original.items()}

_fix_seen_ptrs: dict[int, str] = {}
for k, v in list(sd_fixed.items()):
    ptr = v.data_ptr()
    if ptr in _fix_seen_ptrs:
        sd_fixed[k] = v.clone().contiguous()
    else:
        _fix_seen_ptrs[ptr] = k

_fixed_tmp = Path(tempfile.gettempdir()) / "test_fixed.safetensors"
_fixed_succeeded = False

try:
    _st_save_file(sd_fixed, str(_fixed_tmp))
    _fixed_succeeded = True
    print("  FIXED PATH: safetensors.save_file after generic clone loop → OK")
except RuntimeError as exc:
    print(f"  FIXED PATH: safetensors.save_file STILL FAILED — {exc}")
print()

# ---------------------------------------------------------------------------
# Step 6 — Verify decoder.lm_head.weight is present in the saved file
# ---------------------------------------------------------------------------

print("[Step 6] Verifying decoder.lm_head.weight in saved file …")
print()

_LM_HEAD_KEY = "decoder.lm_head.weight"
_lm_head_present = False

if _fixed_succeeded and _fixed_tmp.exists():
    try:
        with _sf_safe_open(str(_fixed_tmp), framework="pt", device="cpu") as _sf:
            _saved_keys = set(_sf.keys())
        if _LM_HEAD_KEY in _saved_keys:
            _lm_head_present = True
            print(f"  VERIFY: {_LM_HEAD_KEY} present in saved file → OK")
        else:
            print(
                f"  VERIFY: {_LM_HEAD_KEY} MISSING from saved file "
                "→ FAIL (deduplication bug still active)"
            )
    except (OSError, RuntimeError, KeyError) as exc:
        print(f"  VERIFY: could not read saved file — {exc}")
else:
    print("  VERIFY: skipped — fixed-path save did not succeed, nothing to inspect.")
print()

# ---------------------------------------------------------------------------
# Step 7 — Summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("[Step 7] Summary")
print("=" * 70)
print()

_alias_check_pass = True  # always informational
_broken_pass = not _broken_succeeded  # we EXPECT this to fail when aliases exist
_fixed_pass = _fixed_succeeded
_lm_head_pass = _lm_head_present

_rows = [
    ("Alias scan", "INFO", f"{len(_aliases)} alias(es) found"),
    (
        "Broken path (raw SD) fails",
        "PASS" if _broken_pass else "INFO",
        "safetensors correctly raises" if _broken_pass else "No aliases — save succeeded",
    ),
    (
        "Fixed path (clone loop) succeeds",
        "PASS" if _fixed_pass else "FAIL",
        "save_file OK" if _fixed_pass else "save_file raised an error",
    ),
    (
        f"{_LM_HEAD_KEY} in file",
        "PASS" if _lm_head_pass else "FAIL",
        "key present" if _lm_head_pass else "key MISSING — dedup bug",
    ),
]

for label, status, detail in _rows:
    print(f"  [{status:4s}] {label}: {detail}")

print()
_overall_pass = _fixed_pass and _lm_head_pass
if _overall_pass:
    print("  ✅  OVERALL: PASS — generic clone loop correctly de-aliases the state dict")
    print("               and decoder.lm_head.weight is preserved in the saved file.")
else:
    print("  ❌  OVERALL: FAIL — see FAIL rows above for details.")
print()

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

for _tmp in (_broken_tmp, _fixed_tmp):
    try:
        if _tmp.exists():
            _tmp.unlink()
    except OSError:
        pass

sys.exit(0 if _overall_pass else 1)
