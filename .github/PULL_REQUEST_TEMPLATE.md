# Agent Task: Fix dependency resilience in aiparallel0/kaggle

You are a coding agent working on the repository `aiparallel0/kaggle`.
Your task is to apply exactly three targeted fixes described below.
Make all changes in a single commit on a new branch called `fix/dependency-resilience`.
Do not refactor anything beyond what is described. Do not add new tests unless explicitly asked.

---

## Fix 1 — Guard bare PIL imports in `evaluate_models.py` and `dataset_preparation.py`

### Background

`evaluate_models.py` and `dataset_preparation.py` have bare top-level `from PIL import Image`
with no guard. These two files crash on **import** when Pillow is absent, even though every
other file (`train.py`, `train_trocr_yolo.py`, `donut_evaluator.py`) already has a proper
`try/except ImportError` guard + inline PNG/BMP fallback.

### Files to modify

- `evaluate_models.py` — bare `from PIL import Image` at line ~32
- `dataset_preparation.py` — bare `from PIL import Image` at line ~32

### Exact change required

In **both** files, replace the bare import with the following guarded block.
Place it in the same position as the original `from PIL import Image` line:

```python
try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    from donut_evaluator import _load_image, _PIL_AVAILABLE  # noqa: E402, I001

    class _ImageShim:  # type: ignore[misc]
        @staticmethod
        def open(path):
            class _Img:
                def __init__(self, arr):
                    self._arr = arr

                def convert(self, mode):
                    return self

            import numpy as _np  # noqa: F401

            return _Img(_load_image(path))

    Image = _ImageShim()  # type: ignore[assignment]
```

In `evaluate_models.py` the lines immediately after the PIL block are:

```python
from constants import DEVICE, FIELDS, MAX_LENGTH, _get_sroie_dir, _gpu_cleanup
from dataset_loaders import load_sroie_test
```

In `dataset_preparation.py` the lines immediately after the PIL block are:

```python
from constants import FIELDS, IMAGE_EXTS
from dataset_loaders import SROIELoader, _load_key_file
```

### Acceptance criteria

- `python -c "import evaluate_models"` succeeds even when Pillow is not installed
- `python -c "import dataset_preparation"` succeeds even when Pillow is not installed

---

## Fix 2 — Improve `transformers` stub error messages

### Background

`train.py`, `train_trocr_yolo.py`, and `donut_evaluator.py` already have `try/except ImportError`
blocks around `from transformers import ...`, but the `except` branch defines stub classes whose
**only method immediately raises `ImportError`** with a short, unhelpful message.

The error messages should include the exact pip command and point to `requirements.txt`.

### Files to modify and exact changes

#### `train.py`

Find the `VisionEncoderDecoderModel` stub class (in the `except ImportError` block for
transformers, around line 576). It has two methods: `from_pretrained` and
`from_encoder_decoder_pretrained`. Update **both** to use the improved message:

```python
class VisionEncoderDecoderModel:  # type: ignore[no-redef]
    """Placeholder — requires transformers or the inline Swin+BART implementation."""

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):
        raise ImportError(
            "transformers >= 4.37.0 is required but not installed.\n"
            "Run: pip install transformers>=4.37.0\n"
            "Or:  pip install -r requirements.txt"
        )

    @staticmethod
    def from_encoder_decoder_pretrained(*args, **kwargs):
        raise ImportError(
            "transformers >= 4.37.0 is required but not installed.\n"
            "Run: pip install transformers>=4.37.0\n"
            "Or:  pip install -r requirements.txt"
        )
```

#### `train_trocr_yolo.py`

Find the `TrOCRProcessor` and `VisionEncoderDecoderModel` stub classes (in the
`except ImportError` block for transformers, around line 68). Update **both** stubs:

```python
class TrOCRProcessor:  # type: ignore[no-redef]
    @classmethod
    def from_pretrained(cls, *a, **kw):
        raise ImportError(
            "transformers >= 4.37.0 is required but not installed.\n"
            "Run: pip install transformers>=4.37.0\n"
            "Or:  pip install -r requirements.txt"
        )

class VisionEncoderDecoderModel:  # type: ignore[no-redef]
    @classmethod
    def from_pretrained(cls, *a, **kw):
        raise ImportError(
            "transformers >= 4.37.0 is required but not installed.\n"
            "Run: pip install transformers>=4.37.0\n"
            "Or:  pip install -r requirements.txt"
        )
```

#### `donut_evaluator.py`

Find the inner `except ImportError` block (the fallback when both `transformers` and `train.py`
are unavailable, around line 50). Update **both** stub classes:

```python
class DonutProcessor:  # type: ignore[no-redef]
    @classmethod
    def from_pretrained(cls, *a, **kw):
        raise ImportError(
            "transformers >= 4.37.0 is required but not installed.\n"
            "Run: pip install transformers>=4.37.0\n"
            "Or:  pip install -r requirements.txt"
        )

class VisionEncoderDecoderModel:  # type: ignore[no-redef]
    @classmethod
    def from_pretrained(cls, *a, **kw):
        raise ImportError(
            "transformers >= 4.37.0 is required but not installed.\n"
            "Run: pip install transformers>=4.37.0\n"
            "Or:  pip install -r requirements.txt"
        )
```

### Acceptance criteria

- When `transformers` is absent, stub `from_pretrained()` raises `ImportError` with a
  message containing `pip install -r requirements.txt`

---

## Fix 3 — Fix pip auto-installer timeout and error reporting in `run_all.py`

### Background

`_install_dependencies()` in `run_all.py` calls `subprocess.run` with `timeout=300`
(5 minutes). This timeout is too short — `ultralytics` alone can take longer to download
and install on a slow connection. The `TimeoutExpired` exception falls into a bare
`except Exception: pass` block, so the install *appears* to succeed but leaves packages
missing. Additionally, pip stderr is truncated to only 200 characters, hiding the actual
pip error.

### File to modify

`run_all.py` — `_install_dependencies()` function, the `subprocess.run(...)` call and
surrounding error handling.

### Exact changes required

Find this block (around line 235–254):

```python
        try:
            with _InstallWatchdog():
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", tmp_path],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5-minute cap; avoids indefinite hangs
                )
            if result.returncode == 0:
                print(
                    "[setup] Dependencies installed successfully — restarting to load new packages..."
                )
                os.environ["_DONUT_RESTARTED"] = "1"
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                if result.stderr:
                    print(f"[setup] pip warning: {result.stderr[:200]}")
```

Replace it with:

```python
        try:
            with _InstallWatchdog():
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", tmp_path],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30-minute cap; ultralytics + torch can exceed 5 min on slow connections
                )
            if result.returncode == 0:
                print(
                    "[setup] Dependencies installed successfully — restarting to load new packages..."
                )
                os.environ["_DONUT_RESTARTED"] = "1"
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                if result.stderr:
                    print(f"[setup] pip stderr: {result.stderr[:1000]}")
                if result.stdout:
                    print(f"[setup] pip stdout: {result.stdout[:500]}")
        except subprocess.TimeoutExpired:
            print(
                "[setup] TIMEOUT: pip install exceeded 1800s — packages may be partially installed.\n"
                "[setup] Run manually: pip install -r requirements.txt"
            )
```

**Important:** The `except subprocess.TimeoutExpired` block replaces the old implicit fall-through
into the outer `except Exception: pass`. Do NOT remove the outer `except Exception: pass` — it
still serves as the catch-all for other unexpected errors (e.g. `PermissionError`,
`FileNotFoundError`).

### Acceptance criteria

- `run_all.py` `_install_dependencies()` has `timeout=1800`
- `run_all.py` `_install_dependencies()` catches `subprocess.TimeoutExpired` explicitly
  with a printed message containing `pip install -r requirements.txt`
- `run_all.py` prints up to 1000 chars of pip stderr (not 200) on failure
- `run_all.py` also prints up to 500 chars of pip stdout on failure

---

## Files to modify

- `evaluate_models.py`
- `dataset_preparation.py`
- `train.py`
- `train_trocr_yolo.py`
- `donut_evaluator.py`
- `run_all.py`

## Do NOT modify

- `requirements.txt`
- `constants.py`
- Any test file in `tests/`
- Any `.tex` file
- Any file in `paper/`

## After making changes

1. Run `ruff check --fix . && ruff format .` to fix any lint issues
2. Verify: `python -c "import evaluate_models"` exits 0 (even without Pillow installed)
3. Verify: `python -c "import dataset_preparation"` exits 0 (even without Pillow installed)
4. Commit all changes in a single commit with message:
   `fix: guard PIL imports, improve transformers stub errors, fix pip timeout`
