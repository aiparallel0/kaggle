# SOLID Violations & Code Quality Deep-Cleaning Report

> **Generated:** 2026-04-04 | **Scope:** All 14 Python files (34,213 LOC)
> **Status:** Issues marked ✅ were fixed in this PR. Issues marked 🔲 remain as documented technical debt.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Fixed Issues (This PR)](#2-fixed-issues-this-pr)
3. [constants.py](#3-constantspy-549-loc)
4. [data_pipeline.py](#4-data_pipelinepy-3003-loc)
5. [run_experiments.py](#5-run_experimentspy-5081-loc)
6. [train.py](#6-trainpy-4458-loc)
7. [run_all.py](#7-run_allpy-4823-loc)
8. [train_trocr_yolo.py](#8-train_trocr_yolopy-4138-loc)
9. [reporting.py](#9-reportingpy-3446-loc)
10. [cloud_orchestration.py](#10-cloud_orchestrationpy-2564-loc)
11. [validation.py](#11-validationpy-1919-loc)
12. [diagnostics.py](#12-diagnosticspy-1469-loc)
13. [autonomous_ci.py](#13-autonomous_cipy-1036-loc)
14. [resource_manager.py](#14-resource_managerpy-1008-loc)
15. [sweep.py](#15-sweeppy-428-loc)
16. [Remaining Broad Exception Catches](#16-remaining-broad-exception-catches)
17. [Cross-File Issues](#17-cross-file-issues)
18. [Priority Refactoring Roadmap](#18-priority-refactoring-roadmap)

---

## 1. Executive Summary

| Category | Total Found | Fixed | Remaining |
|----------|-------------|-------|-----------|
| Broad `except Exception:` (safe to narrow) | 13 | ✅ 13 | 0 |
| Broad `except Exception:` (unsafe/intentional) | ~48 | 0 | 🔲 48 |
| Duplicate code (regex, imports) | 5 | ✅ 5 | 0 |
| SRP violations (classes/functions) | 22 | 0 | 🔲 22 |
| OCP violations | 8 | ✅ 2 | 🔲 6 |
| LSP violations | 4 | ✅ 1 | 🔲 3 |
| ISP violations | 3 | 0 | 🔲 3 |
| DIP violations | 9 | 0 | 🔲 9 |
| Functions >50 lines | 35+ | 0 | 🔲 35+ |
| Missing return type hints (public) | 58+ | 0 | 🔲 58+ |
| Missing `-> None` on `__init__` | 15+ | ✅ 15 | 0 |
| Bare `list`/`dict` type hints | 146+ | 0 | 🔲 146+ |
| Hardcoded magic numbers | 30+ | ✅ 6 | 🔲 24+ |
| Missing docstrings | 3 | ✅ 3 | 0 |
| Code smells (`__import__()`) | 4 | ✅ 4 | 0 |
| Incomplete type hints (TYPE-1) | 3 | ✅ 3 | 0 |

---

## 2. Fixed Issues (This PR)

### ✅ Exception Narrowing (13 instances across 7 files)

| File | Line | Old | New | Context |
|------|------|-----|-----|---------|
| constants.py | 431 | `except Exception:` | `except (ImportError, RuntimeError):` | Rich progress bar import |
| data_pipeline.py | 54 | `except Exception:` | `except (ImportError, AttributeError):` | Transformers compat shim |
| data_pipeline.py | 507 | `except Exception:` | `except OSError:` | HF token file read |
| data_pipeline.py | 591 | `except Exception:` | `except (KeyError, IndexError, TypeError):` | HF dataset config extraction |
| data_pipeline.py | 605 | `except Exception:` | `except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError):` | HF API size info |
| data_pipeline.py | 1366 | `except Exception: pass` | `except (...) as exc: logger.warning(...)` | FUNSD remap (was silent) |
| sweep.py | 344 | `except Exception:` | `except (json.JSONDecodeError, FileNotFoundError, ValueError):` | Multi-seed JSON load |
| cloud_orchestration.py | 763 | `except Exception:` | `except SyntaxError:` | AST parsing |
| reporting.py | 1728 | `except Exception:` | `except (FileNotFoundError, ValueError, OSError):` | CSV curve read |
| reporting.py | 1943 | `except Exception:` | `except (json.JSONDecodeError, FileNotFoundError, ValueError):` | Selection JSON |
| train_trocr_yolo.py | 213 | `except Exception:` | `except (ValueError, IndexError, MemoryError):` | PPM image parse |
| train_trocr_yolo.py | 1304 | `except Exception:` | `except (TypeError, ValueError, AttributeError):` | Image size config |
| train_trocr_yolo.py | 1317 | `except Exception:` | `except (ImportError, RuntimeError):` | psutil RAM check |
| train_trocr_yolo.py | 1324 | `except Exception:` | `except (FileNotFoundError, ValueError, OSError):` | /proc/meminfo fallback |

### ✅ Duplicate Code Removal

| File | Lines Removed | Issue |
|------|---------------|-------|
| data_pipeline.py | 2263–2272 | Duplicate `_STREET_NUMBER_RE`, `_PO_BOX_RE`, `_STATE_ZIP_RE`, `_STREET_SUFFIX_RE` (already defined at lines 318–332) |
| reporting.py | 1558 | Duplicate `import gc` in same function (already imported at line 1539) |
| cloud_orchestration.py | 1392 | Redundant `import argparse` inside function (already at module-level line 8) |

### ✅ Named Constants Extraction (OCP-1, OCP-5, CROSS-4)

| Constant | File | Old | New |
|----------|------|-----|-----|
| `LABEL_IGNORE_INDEX = -100` | constants.py | Bare `-100` in 6 locations | Named constant, used in constants.py + train.py |
| `BYTE_UNIT_SIZE = 1024` | constants.py | Bare `1024` in format_bytes | Named constant |
| `MAX_DATALOADER_WORKERS = 8` | constants.py | `min(8, ...)` | Named constant |
| `MIN_DATALOADER_WORKERS = 4` | constants.py | `max(4, ...)` | Named constant |
| `DONUT_IMAGE_SIZE = (1280, 960)` | constants.py | `{"height": 1280, "width": 960}` in train.py + run_all.py | Single constant, imported in train.py + run_all.py |

### ✅ Docstrings Added (DOC-1)

| File | Method | Description |
|------|--------|-------------|
| constants.py | `DeduplicatingHandler.emit()` | "Process a log record, collapsing consecutive identical messages." |
| constants.py | `DeduplicatingHandler.flush()` | "Flush any pending deduplicated record, then flush the target handler." |
| constants.py | `DeduplicatingHandler.close()` | "Flush pending records, close the target handler, then close self." |

### ✅ Type Hints Fixed (TYPE-1)

| File | Location | Change |
|------|----------|--------|
| constants.py | `_mask_empty_field_labels()` | Added `labels: "torch.Tensor"`, `tokenizer: "PreTrainedTokenizerBase"` |
| constants.py | `get_disk_usage()` | Replaced string literal types with real `str \| Path \| None` and `tuple[int, int, int]` |
| constants.py | `_progress()` | Added `iterable: Iterable` param type and `-> Generator` return type |

### ✅ `-> None` on `__init__` Methods (CROSS-5)

| File | Class |
|------|-------|
| cloud_orchestration.py | `StorageManager`, `CodeRepairOrchestrator`, `MLTrainingOrchestrator`, `CloudPipelineOrchestrator` |
| reporting.py | `_DonutInference`, `TrOCRYOLOPipeline`, `ResultsAggregator` |
| resource_manager.py | `_AuditLogger` |
| run_all.py | `_DualStreamHandler`, `PipelineOrchestrator` |
| diagnostics.py | `DiagnosticCallback`, `PipelineDiagnostics` |
| train.py | `SROIEOnlyValCallback`, `SROIEDataset`, `MultiDataset`, `DonutTrainer` |

### ✅ `LmHeadCloneCallback` Silent Returns Fixed (LSP-3)

| File | Line | Change |
|------|------|--------|
| train.py | 2649 | Added `logger.debug("...model is None...")` before return |
| train.py | 2652 | Added `logger.debug("...no decoder attr...")` before return |
| train.py | 2660 | Added `logger.debug("...no lm_head.weight...")` else branch |

### ✅ `__import__()` Replaced with Standard Imports (SMELL-1)

| File | Line | Old | New |
|------|------|-----|-----|
| train_trocr_yolo.py | ~1580 | `__import__("logging").getLogger(...)` | `logging.getLogger(...)` |
| train_trocr_yolo.py | ~2583 | `__import__("logging").getLogger(...)` | `logging.getLogger(...)` |
| train_trocr_yolo.py | ~2715 | `__import__("math").cos(...)` | `math.cos(...)` (added `import math` to top) |
| train_trocr_yolo.py | ~2795 | `__import__("logging").getLogger(...)` | `logging.getLogger(...)` |

---

## 3. constants.py (549 LOC)

### 🔲 SRP-1: File mixes 10+ unrelated responsibilities
**Lines:** 1–549 (entire file)
**Severity:** HIGH
**Description:** `constants.py` serves as both a constants file AND a utility library. It contains:
- Configuration constants (lines 45–98)
- Project metadata (lines 458–463)
- Logging handler class `DeduplicatingHandler` (lines 232–321)
- GPU cleanup utility `_gpu_cleanup()` (lines 136–162)
- ML-specific `_mask_empty_field_labels()` (lines 164–206)
- Reproducibility `set_seed()` (lines 209–229)
- Progress bar abstraction `_progress()` (lines 379–449)
- Edit distance metric `_edit_distance()` (lines 328–336)
- File system utilities `format_bytes()`, `get_disk_usage()` (lines 339–377)
- Pipeline validation `validate_pipeline_readiness()` (lines 471–549)

**Recommended split (without breaking imports):**
- Keep all names in `constants.py` via re-exports
- Extract implementations into `utils/logging.py`, `utils/ml.py`, `utils/progress.py`, `utils/metrics.py`, `utils/disk.py`

### ✅ OCP-1: Hardcoded magic numbers — Fixed in PR (2026-04-05)
**Lines:** 133, 165–172, 346–349
**Severity:** MEDIUM
| Line | Value | Replaced with |
|------|-------|---------------|
| 133 | `min(8, max(4, ...))` | `MAX_DATALOADER_WORKERS`, `MIN_DATALOADER_WORKERS` |
| 165–172, 205 | `-100` (label mask ID) | `LABEL_IGNORE_INDEX = -100` |
| 346–349 | `1024` (byte unit threshold) | `BYTE_UNIT_SIZE = 1024` |

### 🔲 DIP-1: `_gpu_cleanup()` tightly coupled to `torch`
**Lines:** 154–162
**Severity:** MEDIUM
**Description:** Direct `import torch` inside function; no abstraction for device cleanup strategy. Hard to test without torch installed.

### 🔲 ISP-1: `validate_pipeline_readiness()` monolithic
**Lines:** 471–549
**Severity:** MEDIUM
**Description:** 90-line function with nested closures. Claims "stdlib-only" but imports `importlib` and `data_pipeline`. Cannot selectively run certain checks. Returns dict structure instead of raising exceptions.

### ✅ DOC-1: `DeduplicatingHandler` methods missing docstrings — Fixed in PR (2026-04-05)
**Lines:** 275, 301, 306
**Severity:** LOW
Added docstrings to `emit()`, `flush()`, `close()`.

### ✅ TYPE-1: Incomplete type hints — Fixed in PR (2026-04-05)
**Lines:** 164 (`labels` param), 353 (string literal types), 379 (`iterable` untyped, missing return type)
**Severity:** LOW
Added `labels: "torch.Tensor"`, `tokenizer: "PreTrainedTokenizerBase"`, real types for `get_disk_usage()`, `_progress(iterable: Iterable) -> Generator`.

---

## 4. data_pipeline.py (3,003 LOC)

### 🔲 OCP-2: Repeated `_dest_dir()`/`_marker()`/`_hf_cache()` methods
**Lines:** 1062–1072, 1235–1245, 1503–1513, 1752–1759
**Severity:** HIGH
**Description:** Four loader classes (`WildReceiptLoader`, `FUNSDLoader`, `InvoicesDonutLoader`, `CORDv2Loader`) each implement identical `_dest_dir()`, `_marker()`, `_hf_cache()` patterns. Should use template method pattern in `BaseDatasetLoader` with abstract `_subdir_name()`.

### 🔲 SRP-2: `_hf_download_dataset_inline()` too long
**Lines:** 522–659 (138 lines)
**Severity:** MEDIUM
**Description:** Single function handles file locks, config discovery, row count fetch, batch download, JSON writing, and marker management. Should decompose into `_hf_discover_config()`, `_hf_fetch_rows_batch()`, `_hf_process_and_save_rows()`.

### 🔲 SRP-3: `get_combined_dataset()` too long
**Lines:** 2155–2244 (119 lines)
**Severity:** MEDIUM
**Description:** Orchestrates dataset loading, SROIE oversampling, 70/15/15 splitting, shuffling, and optional source tracking.

### 🔲 LSP-1: `SROIELoader.clear_cache()` is a no-op
**Lines:** 1010–1012
**Severity:** MEDIUM
**Description:** Base class contract expects `clear_cache()` to clear data, but SROIE returns immediately with a log message. Either make `clear_cache()` non-abstract or document SROIE immutable cache.

### 🔲 DIP-2: `normalise_samples()` instantiates concrete `DatasetNormalizer`
**Lines:** 294–305
**Severity:** MEDIUM
**Description:** Should accept normalizer as optional parameter with default factory.

### 🔲 ISP-2: `get_combined_dataset()` union return type
**Lines:** 2159
**Severity:** LOW
**Description:** Returns `tuple[list, list] | tuple[list, list, list]` depending on `return_sources` flag. Should use `@overload` or return a dataclass.

### 🔲 SRP-4: `CORDv2Loader._cord_remap()` too long
**Lines:** 1809–1902 (94 lines)
**Severity:** MEDIUM
**Description:** Handles multiple CORD-v2 schema variations for all 4 fields. Should extract per-field helpers.

### 🔲 SRP-5: `FUNSDLoader.load()` too long
**Lines:** 1371–1446 (76 lines)
**Severity:** MEDIUM
**Description:** Handles HF Arrow loading, inline JSONL fallback, image conversion, GT remapping. Should split into `_load_from_hf_arrow()` and `_load_from_inline_cache()`.

---

## 5. run_experiments.py (5,081 LOC)

### 🔲 SRP-6: `train_experiment()` — 402-line function
**Lines:** 3741–4143
**Severity:** HIGH
**Description:** Handles hyperparameter loading, model construction, token embedding init, dataset building, OOM recovery, training delegation, and checkpoint saving. Should decompose into 5+ functions.

### 🔲 SRP-7: `run_experiment()` — 378-line function
**Lines:** 4311–4688
**Severity:** HIGH
**Description:** Orchestrates cache checking, disk validation, data loading, config optimization, training, evaluation, cleanup, and result serialization.

### 🔲 SRP-8: `DonutEvaluator` class — 5 responsibilities
**Lines:** 2351–2776
**Severity:** MEDIUM
**Description:** Handles model loading, self-testing, inference, output parsing (SROIE + CORD), and metrics computation. Should split into `ModelLoader`, `Inference`, `Metrics`.

### 🔲 SRP-9: `ExperimentConfig` mixes storage + adapter
**Lines:** 3458–3549
**Severity:** MEDIUM
**Description:** Serves as both training config holder and DonutTrainer interface adapter (duck-typed aliases). Should separate parameter storage from compatibility logic.

### 🔲 OCP-3: Task prompt format hardcoded in 5+ places
**Lines:** 2535, 2692, 2794, 2867, 2876
**Severity:** MEDIUM
**Description:** `if self.task_prompt.startswith("<s_sroie")` and similar checks scattered across methods. Adding a third format requires modifying 5+ locations. Should use strategy pattern.

### 🔲 OCP-4: Parser implementation coupling
**Lines:** 2668–2724
**Severity:** MEDIUM
**Description:** `_parse_prediction()` contains hardcoded branching between SROIE parser and processor.token2json. Not extensible.

### 🔲 LSP-2: `_self_test()` inconsistent return/raise
**Lines:** 2480–2608
**Severity:** MEDIUM
**Description:** Sometimes raises `RuntimeError`, sometimes raises `SelfTestFailedError`, sometimes returns silently. Contract unclear.

### 🔲 DIP-3: Concrete `VisionEncoderDecoderModel` dependencies
**Lines:** 2179, 3801, 3814
**Severity:** MEDIUM
**Description:** Direct dependency on concrete HF model class. Blocks extensibility to other architectures.

### 🔲 DIP-4: Concrete `DonutProcessor` dependencies
**Lines:** 2175, 3795, 3833
**Severity:** MEDIUM
**Description:** Hardcoded dependency on processor class.

### 🔲 TYPE-2: 21+ functions missing return type hints
**Lines:** 454, 494, 659, 1600, 1644, 2175, 2766, 2819, 2910, 2999, 3014, 3082, 3129, 3224, 4144, 4764
**Severity:** MEDIUM
**Key missing:** `load_model_with_tied_weights()`, `run_inference()`, `compute_metrics()`, `normalized_edit_distance()`, `remap_cord_to_sroie()`

### 🔲 TYPE-3: Missing parameter type hints
**Lines:** 2175 (`processor`), 2819 (`model`, `processor`, `preloaded_image`), 2910 (`cord_output`), 2999 (`pred`, `gt`), 3014 (`predictions`, `ground_truths`), 3082 (`pretrained_m`, `finetuned_m`)
**Severity:** MEDIUM

### 🔲 FUNC-1: `_build_model_and_datasets()` — 354 lines
**Lines:** 3790–4143
**Severity:** HIGH

---

## 6. train.py (4,458 LOC)

### 🔲 SRP-10: `DonutTrainer.train()` — 617-line god method
**Lines:** 3528–4145
**Severity:** HIGH (CRITICAL)
**Description:** Single method handles: precision detection, warmup capping, parameter grouping, callback registration (5+ callbacks), DataLoader config, LR scheduler creation, 6 pre-training guardrails, gradient checkpointing, dataset patching, custom trainer instantiation, training loop, worker cleanup. Should decompose into `_configure_precision()`, `_build_optimizer()`, `_register_callbacks()`, `_validate_config()`, `_execute_training()`.

### 🔲 SRP-11: `LiveDashboardCallback` — 294 lines, mixed concerns
**Lines:** 3168–3462
**Severity:** MEDIUM-HIGH
**Description:** Manages CSV I/O, renders rich tables, queries system metrics (VRAM/disk), tracks best F1, handles lifecycle. Should split into `MetricsCSVWriter`, `TrainingMetricsTracker`, `RichTableRenderer`.

### 🔲 SRP-12: `MultiDataset.__init__()` — 147-line constructor
**Lines:** 2942–3088
**Severity:** HIGH
**Description:** Single `__init__` handles validation, RAM estimation, concurrent image loading, tensor precomputation, label precomputation, and logging (11 logger calls). Should use builder pattern.

### 🔲 SRP-13: `SROIEOnlyValCallback` — mixed calculation + I/O + logging
**Lines:** 2718–2820
**Severity:** MEDIUM
**Description:** F1 calculation, CSV write, logging, and model inference all in one callback.

### ✅ OCP-5: Hardcoded image dimensions — Fixed in PR (2026-04-05)
**Lines:** 453, 2981
**Severity:** MEDIUM
**Description:** Replaced with `DONUT_IMAGE_SIZE` constant from `constants.py`.

### 🔲 OCP-6: `LiveDashboardCallback.__init__()` monkey-patching
**Lines:** 3190–3200
**Severity:** MEDIUM
**Description:** Dynamic class type replacement via `self.__class__ = type(...)`. Prevents clean extension.

### ✅ LSP-3: `LmHeadCloneCallback.on_save()` silent early returns — Fixed in PR (2026-04-05)
**Lines:** 2647–2660
**Severity:** MEDIUM
**Description:** Added debug-level logging to all early returns so skipped clones are observable.

### 🔲 DIP-5: `DonutTrainer` — direct concrete dependencies
**Lines:** 3470–4279
**Severity:** HIGH
**Description:** Direct `Seq2SeqTrainer` instantiation, direct `torch.optim.AdamW`/`SGD` creation, hardcoded callback classes.

### 🔲 DIP-6: `MultiDataset` — processor dependency
**Lines:** 2942–3152
**Severity:** MEDIUM
**Description:** Direct `DonutProcessor` type dependency. No interface abstraction.

### 🔲 EXCEPT-BROAD: 23 instances of `except Exception`
**Lines:** 657, 669, 2211, 2266, 2438, 2795, 2798, 2818, 2988, 3042, 3244, 3322, 3391, 3401, 3461, 3659, 3828, 3861, 3887, 3898, 3980, 4268, 4269
**Severity:** HIGH (many mask real bugs, especially in callbacks)

### 🔲 FUNC-2: `_load_jpeg_pure` — 460 lines
**Lines:** 2032–2491
**Severity:** HIGH (pure Python JPEG decoder; extremely complex)

### 🔲 FUNC-3: `save()` method — 99 lines
**Lines:** 4172–4270
**Severity:** MEDIUM

---

## 7. run_all.py (4,823 LOC)

### 🔲 SRP-14: `PipelineOrchestrator` — too many responsibilities
**Lines:** 3247–3481
**Severity:** HIGH
**Description:** Manages stage orchestration, exception handling with AI diagnostics, timing, result aggregation, pipeline diagnostics, and console formatting.

### 🔲 OCP-7: Hardcoded stage list in `PipelineOrchestrator.run()`
**Lines:** 3341–3412
**Severity:** HIGH
**Description:** Multiple `if` statements checking skip flags. Adding a new stage requires modifying `run()`, adding args to parser (4253–4586), updating `stage_labels` dict (3443–3454), and updating `__repr__`.

### 🔲 OCP-8: Six mode handlers with duplicated patterns
**Lines:** 3482–4055
**Severity:** MEDIUM
**Description:** `_quick_mode_handler` (79 lines), `_quick_all_mode_handler` (128 lines), `_mini_mode_handler` (81 lines), `_micro_mode_handler` (138 lines), `_superfast_mode_handler` (71 lines), `_instant_mode_handler` (77 lines) — similar patterns repeated. Should use factory pattern.

### 🔲 DIP-7: `stage_experiments()` — concrete module imports
**Lines:** 1785–2104
**Severity:** MEDIUM
**Description:** Direct `import run_experiments as re_mod`, `DonutProcessor.from_pretrained()` inside function.

### 🔲 TYPE-4: Untyped `args` parameter across 10+ functions
**Lines:** 1024, 1528, 2105, 2138, 2191, 3250, 4056
**Severity:** MEDIUM
**Description:** `args` parameter has no type hint in many functions. Should be `argparse.Namespace` or a dataclass.

### 🔲 EXCEPT-BROAD: 22 instances of `except Exception`
**Lines:** 522, 579, 683, 700, 724, 753, 812, 891, 944, 955, 1001, 1008, 1018, 1051, 1057, 1141, 1325, 1566, 1578, 1591, 1599, 1608
**Severity:** MEDIUM

### 🔲 FUNC-4: `build_parser()` — 334 lines
**Lines:** 4253–4586
**Severity:** MEDIUM

### 🔲 FUNC-5: `stage_experiments()` — 320 lines
**Lines:** 1785–2104
**Severity:** MEDIUM

### 🔲 FUNC-6: `main()` — 238 lines
**Lines:** 4587–4824
**Severity:** MEDIUM

### 🔲 GLOBAL-1: Mutable global `_FANCY_OUTPUT`
**Lines:** 667
**Severity:** MEDIUM
**Description:** Modified at runtime based on `--fancy` flag, accessed in multiple functions. Tests or concurrent runs will interfere.

---

## 8. train_trocr_yolo.py (4,138 LOC)

### 🔲 SRP-15: `TrOCRReceiptDataset` — 21 methods, god class
**Lines:** 1216–3470 (estimated)
**Severity:** HIGH (CRITICAL)
**Description:** Handles data loading, tensor caching, augmentation, preprocessing. `__init__` is 157 lines. Should split into `DatasetLoader`, `CacheManager`, `AugmentationPipeline`.

### 🔲 FUNC-7: `_lr_lambda` — 401 lines
**Lines:** Varies (estimated)
**Severity:** HIGH (EXTREME length)

### 🔲 FUNC-8: `train_trocr` — 324 lines
**Severity:** HIGH

### 🔲 FUNC-9: `evaluate_trocr_yolo_on_test` — 226 lines
**Severity:** HIGH

### 🔲 FUNC-10: `train_field_assigner` — 178 lines
**Severity:** MEDIUM

### ✅ SMELL-1: 5 `__import__()` misuses — Fixed in PR (2026-04-05)
**Lines:** ~1580, ~2583, ~2715 (×2), ~2795
**Severity:** MEDIUM
**Description:** Replaced `__import__("math")` and `__import__("logging")` with top-level `import math` and standard `logging.getLogger(__name__)` calls.

### 🔲 EXCEPT-BROAD: 19+ instances of `except Exception`
**Lines:** ~1454, ~1640, ~2192, ~2197, ~2329, ~3037, ~3041, ~3045, ~3049, ~3053, ~3057, ~3061, ~3065
**Severity:** MEDIUM

---

## 9. reporting.py (3,446 LOC)

### 🔲 SRP-16: `TrOCRYOLOPipeline` — 33 methods, god class
**Lines:** ~600–2700
**Severity:** HIGH (CRITICAL)
**Description:** Handles data loading, result plotting, PDF generation, benchmarking, field assignment. Should split into `DataProcessor`, `MetricsRenderer`, `PDFBuilder`, `BenchmarkReporter`.

### 🔲 SRP-17: `PaperInjector` — 27 methods, god class
**Lines:** ~2000–3200
**Severity:** HIGH
**Description:** Handles PDF editing, metrics compilation, figure rendering. Should split into `PDFEditor`, `MetricsAggregator`, `FigureRenderer`.

### 🔲 EXCEPT-BROAD: 8 instances of `except Exception`
**Lines:** ~2332, ~2344, ~2384, ~2422, ~2723, ~3263
**Severity:** MEDIUM
**Description:** Silent failures in PDF compilation cascade into errors.

### 🔲 DEAD-1: `_YOLO_CLS` potentially dead code
**Lines:** ~784
**Severity:** LOW

### 🔲 SMELL-2: Hardcoded `sys.path` manipulation
**Lines:** 27–29
**Severity:** MEDIUM
**Description:** Fragile assumption about module location.

---

## 10. cloud_orchestration.py (2,564 LOC)

### 🔲 SRP-18: Two monolithic `__init__` methods
**Lines:** 2222–2404 (`CloudPipelineOrchestrator`, 183 lines), 2405–2564 (`CodeRepairOrchestrator`, 160 lines)
**Severity:** HIGH
**Description:** Should use builder pattern or factory functions.

### 🔲 DIP-8: Hard-coded subprocess/git/storage path dependencies
**Severity:** MEDIUM
**Description:** Direct subprocess calls prevent testability.

### 🔲 DUP-1: Two similar `to_dict()` methods
**Lines:** ~122 lines and ~118 lines
**Severity:** MEDIUM
**Description:** Should use `dataclasses.asdict()` or a mixin.

---

## 11. validation.py (1,919 LOC)

### 🔲 SRP-19: `CheckpointValidator.__init__` — 383 lines
**Lines:** ~1160–1542
**Severity:** HIGH (EXTREME)
**Description:** Single `__init__` handles metrics init, state setup, config, checkpoint verification. Should use builder pattern: `ConfigValidator`, `MetricsInitializer`, `CheckpointVerifier`.

---

## 12. diagnostics.py (1,469 LOC)

### 🔲 SRP-20: `DiagnosticCallback` — 22 methods, god class
**Lines:** ~174 onwards
**Severity:** HIGH
**Description:** Should split into `TrainingMonitor`, `PatternDetector`, `TelemetryCollector`, `Reporter`.

### 🔲 TYPE-5: 12+ functions without return type hints
**Lines:** 212, 218, 287 (callback methods), and others
**Severity:** LOW

### 🔲 EXCEPT-BROAD: 2 instances of `except Exception`
**Lines:** ~354, ~672
**Severity:** LOW

---

## 13. autonomous_ci.py (1,036 LOC)

### 🔲 FUNC-11: `auto_fix_and_retry()` — 251 lines
**Lines:** ~786–1036
**Severity:** HIGH
**Description:** Multiple nested loops, API calls, file operations in single function. Should break into `run_test_once()`, `analyze_failure()`, `apply_fix()`, `retry_loop()`.

### 🔲 SRP-21: `TestSuiteResult` — 16 methods, god class
**Severity:** MEDIUM
**Description:** Should split into `TestResult`, `ResultParser`, `ResultFormatter`, `DiffGenerator`.

---

## 14. resource_manager.py (1,008 LOC)

### 🔲 FUNC-12: `optimize_hyperparams()` — 218 lines
**Lines:** Various
**Severity:** MEDIUM
**Description:** Monolithic. Should split into `ParamValidator`, `RAMCalculator`, `HyperparamOptimizer`.

### 🔲 FUNC-13: `_get_total_ram_bytes()` — 83 lines
**Severity:** LOW

---

## 15. sweep.py (428 LOC)

**Status:** ✅ Clean — No major violations. Well-structured.

---

## 16. Remaining Broad Exception Catches

These `except Exception:` blocks were **NOT narrowed** because they:
- Are intentional (logging handlers MUST NOT propagate errors)
- Guard against unpredictable library behavior (e.g., torch internal errors)
- Serve as OOM recovery boundaries
- Wrap complex third-party library calls with many possible error types

**Intentional/Correct (do NOT narrow):**
| File | Line | Reason |
|------|------|--------|
| constants.py | 295 | `DeduplicatingHandler.emit()` — logging handlers must catch all |
| train.py | 2795, 2798 | OOM recovery in training callbacks |
| train.py | 3980 | Gradient checkpointing fallback |
| run_all.py | 522, 579 | Logging handler emit (same pattern) |

**Should narrow but complex (defer to dedicated PR):**
All instances in train.py (23), run_all.py (22), train_trocr_yolo.py (19+), reporting.py (6+), diagnostics.py (2) — total ~72 instances. Each requires careful analysis of the try block's possible exceptions.

---

## 17. Cross-File Issues

### 🔲 CROSS-1: No abstract interface for model/processor
**Files:** run_experiments.py, train.py, data_pipeline.py, reporting.py
**Description:** All files directly depend on concrete `VisionEncoderDecoderModel` and `DonutProcessor`. No protocol or ABC exists for swapping architectures.

### 🔲 CROSS-2: `args` parameter untyped everywhere
**Files:** run_all.py, run_experiments.py, reporting.py
**Description:** `argparse.Namespace` passed as untyped `args`. Should define a dataclass or TypedDict for the namespace.

### 🔲 CROSS-3: Duplicate lazy imports of `torch`, `numpy` across files
**Files:** constants.py, train.py, diagnostics.py, train_trocr_yolo.py
**Description:** Same module imported lazily inside multiple functions. Consider top-level conditional import.

### ✅ CROSS-4: Image dimensions `(1280, 960)` hardcoded in multiple files — Fixed in PR (2026-04-05)
**Files:** train.py (lines 453, 2981), run_all.py (lines 1630, 1633), train_trocr_yolo.py
**Description:** Defined `DONUT_IMAGE_SIZE = (1280, 960)` in `constants.py` and used it in train.py and run_all.py.

### ✅ CROSS-5: Missing `-> None` on 15+ `__init__` methods — Fixed in PR (2026-04-05)
**Files:** cloud_orchestration.py (4), reporting.py (3), diagnostics.py (2), resource_manager.py (1), run_all.py (2), train.py (4)
**Severity:** LOW but pervasive
Added `-> None` return type annotation to 16 `__init__` methods across 6 files.

---

## 18. Priority Refactoring Roadmap

### P0 — Critical (Affects maintainability and testability)
1. **Split `DonutTrainer.train()`** (train.py:3528–4145, 617 lines) into 5+ methods
2. **Split `train_experiment()`** (run_experiments.py:3741–4143, 402 lines)
3. **Split `run_experiment()`** (run_experiments.py:4311–4688, 378 lines)
4. **Split `auto_fix_and_retry()`** (autonomous_ci.py:786–1036, 251 lines)
5. **Extract `TrOCRReceiptDataset`** god class (train_trocr_yolo.py:1216+)

### P1 — High (OCP violations blocking extensibility)
6. **Template method for dataset loaders** (data_pipeline.py — 4 classes × 3 methods)
7. **Strategy pattern for task prompt parsers** (run_experiments.py — 5+ locations)
8. **Factory pattern for mode handlers** (run_all.py — 6 handlers)
9. **Stage registry** to replace hardcoded stage list (run_all.py:3341–3412)

### P2 — Medium (Type safety and documentation)
10. **Add return type hints** to 58+ public functions
11. **Add `-> None`** to 15+ `__init__` methods
12. **Parameterize `list`/`dict` type hints** (146+ instances)
13. **Define `DONUT_IMAGE_SIZE` constant** and use across all files
14. **Extract magic numbers** to named constants (30+ instances)

### P3 — Low (Style and cleanup)
15. **Consolidate duplicate lazy imports** across files
16. **Replace `__import__()` calls** with standard imports (train_trocr_yolo.py)
17. **Add docstrings** to `DeduplicatingHandler` methods
18. **Define `args` TypedDict** for CLI argument passing

---

*This document is a living inventory. When resolving an issue, change its 🔲 to ✅, add the PR number and date (e.g., `✅ Fixed in PR #200 (2026-04-05)`), and move it to section 2 ("Fixed Issues"). Do not delete entries — they serve as an audit trail.*
