# Guardrail Gap Analysis: TrOCR+YOLO Pipeline

This document records every guardrail in the TrOCR+YOLO pipeline, what each one
actually checks, what it misses, and which specific bugs slip through the gap.
All citations reference file paths and line numbers in the repository at the time
of analysis (2026-04-03).

---

## 1. `validate_training_config` — The "955 Steps OK" False Assurance

**Location:** `resource_manager.py:706–754`

**Full implementation (body):**
```python
steps_per_epoch = math.ceil(num_train_samples / (batch_size * gradient_accumulation_steps))
total_steps = steps_per_epoch * epochs
if total_steps < min_optimizer_steps:   # default 200
    raise ValueError(...)
logger.info("[validate_training_config] OK: %d optimizer steps ...", total_steps)
```

**What it checks:** One arithmetic inequality —
`ceil(N / (batch × accum)) × epochs ≥ 200`. That is the entire body.

**What it does NOT check:**
- Optimizer type (AdamW vs SGD vs any other)
- Learning rate magnitude or sanity range
- LR scheduler compatibility with actual step count
- Model architecture consistency
- Dataset label alignment
- Whether GradScaler-skipped steps corrupt the warmup budget

**Bug that slips through:**  
The TrOCR inline training loop (`train_trocr_yolo.py:2029–2035`) calls
`scheduler.step()` **unconditionally** after every optimizer step, regardless of
whether GradScaler detected overflow and skipped that step:

```python
# train_trocr_yolo.py:2029–2035
if (step + 1) % grad_accum == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()   # ← unconditional — wastes warmup budget on skipped steps
    optimizer.zero_grad()
```

`validate_training_config` reports "OK: 955 steps" — but each GradScaler-skipped
step burns a warmup token without updating weights. The effective LR curve diverges
from actual weight-update count. No guardrail catches this.

Compare with the inline YOLO loop which correctly guards the scheduler step
(`train_trocr_yolo.py:881–885`):

```python
if scaler.get_scale() < _scale_before:
    skipped_steps += 1
else:
    scheduler.step()   # ← only fires when weights were actually updated
```

This inconsistency exists inside the same file.

---

## 2. "Pre-training guardrails PASSED: tie_word_embeddings=False" — The Core Deduplication Gap

**Location:** `train_trocr_yolo.py:1879–1912`

Three guardrails checked at training start:
- **Guardrail A** (line 1881–1886): `model.config.decoder_start_token_id is not None`
- **Guardrail B** (line 1887–1893): `decoder_start_token_id ≠ unk_token_id`
- **Guardrail C** (line 1895–1912): `tie_word_embeddings is not True` on both
  `model.config` and `model.decoder.config`

**Success log (line 1909–1912):**
```python
print(f"  [TrOCR] Pre-training guardrails PASSED: tie_word_embeddings=False, "
      f"decoder_start_token_id={_trocr_dst_id}")
```

**Why the lm_head WARNING can still fire despite this PASSED message:**

`tie_word_embeddings=False` only tells HuggingFace **not to call `tie_weights()`
on checkpoint reload**. It has no effect on safetensors serialization behavior.

safetensors deduplicates tensors based on `data_ptr()` equality — pointer identity,
not the config flag. After `from_pretrained()`, `lm_head.weight` and
`embed_tokens.weight` may share the same underlying storage even when
`tie_word_embeddings=False`.

The pipeline breaks this alias at load time (lines 1663–1674) and again before each
best-checkpoint save (lines 2108–2111). But the integrity check that verifies the
shard after save (`train_trocr_yolo.py:2117–2138`) is wrapped in a bare
`except Exception`:

```python
try:
    ...parse safetensors header...
    if _trocr_lm_head_key not in _hdr:
        print(f"  [TrOCR] WARNING: {_trocr_lm_head_key} is missing ...")  # print, not raise
    else:
        print(f"  [TrOCR] Checkpoint OK: ...")
except Exception as _cke:
    print(f"  [TrOCR] Checkpoint integrity check skipped: {_cke}")   # silent non-fatal
```

If the header parse fails for any reason (safetensors version mismatch, file lock,
wrong path), the integrity check is silently skipped. The WARNING on a missing key
is only a `print()`, not a `raise` — training continues and an unusable model is
saved without any hard failure.

**The false sense of security:** "Pre-training guardrails PASSED" checks the config
flag once at training start. The deduplication bug fires on tensor storage identity
at save time, not at config read time.

---

## 3. GradScaler / AMP Skip — No Threshold or Abort for TrOCR

**Inline YOLO training** (`train_trocr_yolo.py:907–916, 942–947`): correctly warns
per epoch with count/total ratio, warns at training end if any skips occurred, and
only steps the scheduler when the optimizer step was not skipped.

**TrOCR training** (`train_trocr_yolo.py:2029–2035`): `scheduler.step()` is called
unconditionally. There is no per-step or per-epoch count of skipped steps in the
TrOCR loop. There is no aggregate warning. There is no abort threshold.

The NaN-epoch abort (`train_trocr_yolo.py:2071–2083`) requires 3 **consecutive**
full epochs of NaN average loss. A scenario where 30–40 % of steps are skipped
every epoch (gradients overflowing but recoverable) is not caught by the batch
failure counter because GradScaler skips are **not** exceptions — they are silent
optimizer no-ops.

**Gap:** There is no check that computes `(skipped steps / total steps)` for TrOCR
and warns if it exceeds a threshold (e.g., 10 %).

---

## 4. YOLO State Dict Loading — `strict=False` and the Architecture Mismatch Silence

**Location:** `train_trocr_yolo.py:590–614`

```python
missing, unexpected = self.model.load_state_dict(sd, strict=False)
self._log.info(
    "Loaded YOLOv8x from %s (missing=%d, unexpected=%d)",
    sd_path, len(missing), len(unexpected),
)
```

**What "missing=0, unexpected=0" actually means:**

The inline `_YOLO_CLS` always builds `_YOLOv8Model(nc=1)` (line 579), hardcoded to
the YOLOv8x architecture. The companion state dict `best_sd.pt` is saved by
`torch.save(self.model.state_dict(), ...)` (line 940) — also from `_YOLOv8Model`.
When the inline fallback is used end-to-end, the architectures match and `missing=0`
is genuine.

**Critical gap:** `strict=False` means `missing` and `unexpected` are logged as
information only. No exception is raised regardless of count. If architectures
diverge (e.g., a real `yolov8n` checkpoint placed where `yolov8x` is expected),
there is no hard failure.

The real problem is that the inline fallback trains a **proxy loss** (line 869):
```python
loss = sum((f.float().pow(2).mean() - 1.0).pow(2) for f in raw_feats)
```
This is a unit-energy regularization, not text-box detection. The comment at lines
858–867 is explicit: *"Full YOLO detection loss (box regression + DFL +
classification) requires the ultralytics TaskAlignedAssigner and is not inlined
here."*

"Loaded YOLOv8x from best_sd.pt (missing=0, unexpected=0)" confirms key alignment.
It says nothing about whether the weights encode meaningful text-detection
capability.

**Gap:** There is no check at inference time (in `_extract_ocr_lines`) for:
- Number of boxes detected per image (zero boxes → empty `ocr_lines` → all
  fields empty → F1 = 0 with no error)
- Confidence score distribution (proxy-trained YOLO produces low-confidence boxes
  that may be filtered by `score_thr=0.25` in `_nms_boxes`)
- Whether detected boxes align with receipt layout

---

## 5. Field Coverage Checks — Data-Side Only, No Output-Side Check

The log messages like `"company = 63/63 (100.0%)"` appear in `run_all.py`'s
`_SUPPRESS_SUBSTRINGS` list (line 375):
```python
"field coverage:",   # suppressed to file-only
```
These are written to `terminal.txt` but never to console. They verify that SROIE
ground-truth JSON has values for each of the 4 fields.

**What they check:** Input data quality — GT labels are non-empty.

**What they do NOT check:** Whether the YOLO detector actually produces boxes that,
when fed through TrOCR and field assignment, result in non-empty output for each
field.

In `_extract_ocr_lines` (`train_trocr_yolo.py:2902`):
```python
if yolo_results and len(yolo_results[0].boxes) > 0:
    # ... process boxes ...
return ocr_lines, ...
```
If YOLO detects zero boxes, `ocr_lines` is returned as `[]`.
`_assign_fields_heuristic([])` at line 2998 returns `{}`. All fields score as
false negatives. There is no warning that YOLO detected zero boxes. This is the
structural reason `address_f1 = 0.000` appears in inline-fallback runs — it is a
silent empty result from the detection stage, not an exception.

---

## 6. The Evaluation Loop — Silent Failure Path

**`evaluate_trocr_yolo_on_test` (`run_experiments.py:3323–3329`):**
```python
with torch.no_grad():
    for img_path, _gt in _progress(test_samples, desc="TrOCR+YOLO eval"):
        pred = run_pipeline(img_path, yolo_model, trocr_model, trocr_processor)
        predictions.append(pred)
```
No per-image exception handler. Any exception propagates and fails the entire
evaluation call.

**`_evaluate_field_assigner` (`run_all.py:2486–2496`):**
```python
try:
    pred = _tty.run_trocr_yolo_inference(...)
except Exception as _exc:
    logger.debug("Inference failed for %s: %s", img_path, _exc)  # DEBUG = file-only
    pred = {f: "" for f in FIELDS}
```
All inference exceptions are caught at **DEBUG level** (file-only per
`_DualStreamHandler`). Every failed image silently contributes an empty-prediction
dict that scores as all false negatives. The operator sees a low F1 number with no
indication of how many images failed entirely vs. how many produced wrong
predictions.

**Stage-level catch (`run_all.py:2446–2453`):**
```python
except Exception as exc:
    traceback.print_exc()
    _log.warning("TrOCR+YOLO stage failed: %s: %s", type(exc).__name__, exc)
    return StageResult(name="TrOCR+YOLO", duration=0.0, exit_status=1, warnings=warnings)
```
The entire stage is swallowed as a warning. `exit_status=1` is set but the outer
pipeline continues. There is no F1 = 0 check equivalent to the one applied to DONUT
experiments (`run_all.py:2014` via `--verify`). The `--verify` flag only checks
DONUT results.

---

## 7. The lm_head Deduplication Bug — Exact Failure Path

Three steps to failure:

1. `model.config.tie_word_embeddings = False` is set (line 1642). "Pre-training
   guardrails PASSED" is logged (line 1909). ✓ Config flag is correct.

2. The alias is broken immediately after load (lines 1663–1674). lm_head.weight is
   now independent. ✓

3. **The unguarded gap:** During training, certain in-place optimizer operations can
   cause the alias to be re-established. The clone-before-save at lines 2108–2111
   is designed to catch this — but it only fires when `avg_val < best_val_loss`.
   If the very first epoch produces the best validation loss and the `hasattr` chain
   at line 2108 returned `False` (e.g., due to a model-wrapping mismatch), the
   saved checkpoint would be missing `lm_head.weight`.

The integrity check at lines 2117–2138 is the true last line of defense, but it:
1. Issues only a `print()` warning, not a `raise`
2. Is silently skipped on any exception
3. Only inspects the key name in the safetensors header, not the weight values

---

## 8. Overall Pattern — False Sense of Security

| Log Message | File | Line | What It Actually Checks | What the Bug Path Bypasses |
|---|---|---|---|---|
| `[validate_training_config] OK: 955 optimizer steps` | resource_manager.py | 745 | `N / (batch×accum) × epochs ≥ 200` | Scheduler.step() fires on GradScaler-skipped steps in TrOCR loop |
| `[TrOCR] Pre-training guardrails PASSED: tie_word_embeddings=False` | train_trocr_yolo.py | 1909 | Config flag ≠ True; IDs non-null | safetensors deduplication by `data_ptr()`, not config flag |
| `Loaded YOLOv8x from best_sd.pt (missing=0, unexpected=0)` | train_trocr_yolo.py | 591–596 | State dict keys match model keys | Weights trained with proxy L2 loss, not real detection loss |
| `[TrOCR] Checkpoint OK: decoder.lm_head.weight present in shard` | train_trocr_yolo.py | 2133–2136 | Key name exists in safetensors header | Weight value correctness; silently skipped on exception |
| `[inline-YOLO] Epoch N: GradScaler skipped X/Y steps` | train_trocr_yolo.py | 907–916 | Per-epoch skip count for YOLO | No threshold triggers abort; TrOCR loop has no equivalent check |
| `company = 63/63 (100.0%)` (file-only) | run_all.py | 375 | GT labels are non-empty | YOLO detecting 0 boxes → all predictions empty → F1=0, no warning |
| Stage warning `TrOCR+YOLO stage failed` | run_all.py | 2450 | Stage-level exception caught | Pipeline continues; no F1=0 check applied to TrOCR results |

**The meta-pattern:** Every "OK" / "PASSED" / "complete" message is honest about
what it checked. Each checks exactly one narrow condition (config flag, arithmetic,
key name, data size). The actual failure paths live between these checkpoints — in
the spaces the guardrails do not cover:

- Between the pre-training config check and the save-time tensor identity check
- Between the YOLO "missing=0" load check and per-image box quality at inference
- Between the GradScaler skip warning and the scheduler's consumed warmup budget
- Between the data-side field coverage check and the output-side empty-box result

The pipeline can traverse this entire path — all guardrails reporting success —
while every TrOCR prediction is empty due to zero YOLO box detections, and the
resulting F1 = 0 per field is reported as a valid result with no error or alert.
