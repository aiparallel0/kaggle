# FORENSIC AUDIT — CONDENSED STRIKE PROMPT
> **Standard:** MIT-licensed, OOP-refactored, production-grade. Every output must be real, labeled, and reproducible.
> **Minimum code delta:** 3,800 lines of substantive change across 8 files. Cosmetic edits do not count.
> **Evidence base:** Live training log from 2026-02-23 run + FORENSIC_AUDIT_PROMPT_V2_FILLED.md

---

## SECTION 0 — WHAT THE LIVE RUN PROVED (READ THIS FIRST)

The training run of 2026-02-23 12:20–ongoing produced the following confirmed evidence.
Every issue below was observed in the actual log, not theorized.
Do not dismiss any of these as "might be fine." They are not fine.

**CONFIRMED FAILURE — Stage 1.5 (Pretrained Baseline)**
The checkpoint `naver-clova-ix/donut-base-finetuned-cord-v2` scored **F1 = 0.1533** on its own benchmark dataset (CORD test).
Published benchmark for this exact checkpoint is >0.90. The gap is not noise. It is ~0.75 F1 points.
This is a complete evaluation pipeline failure. The evaluate.py metric computation, token2json parsing, or answer normalization is broken.
Every downstream F1 number from every experiment is therefore untrustworthy until this is diagnosed and fixed.
This is the single most important finding in this audit. Fix it before touching anything else.

**CONFIRMED FAILURE — Experiment 2 (F1 = 0)**
Experiment 2 completed training (1,483 seconds, 1,512 samples), saved a model, ran evaluation on 63 SROIE test images, and reported **Global F1 = 0**.
Not 0.01. Not low. Zero. The model produced no correct predictions by any measure.
The log shows `{'train_loss': 0.7065, 'epoch': 9.0}` — training did not diverge. This is an inference or evaluation bug, not a training failure.
Most likely cause: `decoder.lm_head.weight` missing keys warning (`There were missing keys in the checkpoint model loaded: ['decoder.lm_head.weight']`) — the model's language model head was not loaded at inference time, producing random token sequences that parse to nothing.

**CONFIRMED ANOMALY — Interleaved Log Streams**
Starting at approximately Experiment 1 epoch 21 in the log, training loss sequences from two different experiments are interleaved in the same stdout stream.
One stream shows Experiment 1 validation loss reaching 0.0013 (healthy, near-converged).
The other shows eval_loss of 1.005 → 1.053 → 1.121 → 1.142 → 1.195 rising monotonically over epochs 5–9 (active divergence) while training loss is simultaneously falling (0.21 → 0.09).
This is not oscillation. This is a train/validation distribution mismatch or data contamination in whatever experiment produced the second stream.
The three-progress-bar output confirms multiple trainers were running concurrently, which invalidates GPU memory estimates and timing data for all affected experiments.

**CONFIRMED ANOMALY — Eval Runtime Spikes**
Experiment 1 epoch 16 eval_runtime = 8.66s vs. typical 3.0s. Epoch 17 = 7.05s. All other epochs ~3s.
This suggests GC pressure, memory swapping, or a competing process. In a single-GPU environment, this is likely the concurrent experiment above.
This is a systemic problem with run_all.py's orchestration architecture.

**CONFIRMED WARNING — decoder.lm_head.weight Missing**
The log contains: `There were missing keys in the checkpoint model loaded: ['decoder.lm_head.weight']`
This appears at the end of Experiment 2's training and is directly followed by F1 = 0.
This key is not optional. The LM head produces the output token distribution. A model without it cannot generate any meaningful sequence.
The checkpoint save/load logic in train.py is either saving an incomplete state dict or loading with an incompatible config.

---

## SECTION 1 — ARCHITECTURE MANDATE (OOP REFACTOR REQUIRED)

The current codebase is procedural: loose functions in flat modules with no encapsulation, no interface contracts, and no ownership of state.
This must be refactored to an OOP architecture. The following class structure is required. This is not optional.

**Minimum required class hierarchy:**

```
BaseDatasetLoader (ABC)                  — dataset_loaders.py
  ├── SROIELoader(BaseDatasetLoader)
  ├── WildReceiptLoader(BaseDatasetLoader)
  ├── SROIENERLoader(BaseDatasetLoader)
  ├── CORDLoader(BaseDatasetLoader)
  └── InvoicesDonutLoader(BaseDatasetLoader)

  Each loader must implement:
    load(split: str) -> List[Dict]         # raises DatasetLoadError, never returns []
    validate_cache() -> bool               # checks schema, not just file presence
    clear_cache() -> None
    sample_count(split: str) -> int

ExperimentConfig (dataclass)             — run_experiments.py
  fields: name, datasets, epochs, lr, batch_size, seed
  must be the SINGLE source of truth — no duplicate configs in train.py

DonutTrainer                             — train.py
  __init__(config: ExperimentConfig, processor, model)
  train() -> TrainingResult
  save(path: Path) -> None
  Zero-sample guard in __init__: raises ValueError if dataset is empty

DonutEvaluator                           — evaluate.py
  __init__(model_path: Path, processor, test_dataset)
  evaluate() -> EvaluationResult          # raises if F1 computation is broken
  _parse_prediction(tokens: str) -> Dict  # guarded, returns {} on failure WITH log
  _compute_f1(preds, labels) -> float     # unit-testable, documented behavior on empty preds
  _compute_ned(preds, labels) -> float    # unit-testable, lower-is-better confirmed

PaperInjector                            — inject_results.py
  __init__(results_dir: Path, template_path: Path)
  build_var_map() -> Dict[str, str]       # no hardcoded experiment results
  fill() -> str                           # raises if any \VAR{} remains unresolved
  verify_leaderboard_scores() -> None     # asserts against known-good values

PipelineOrchestrator                     — run_all.py
  stages run SEQUENTIALLY, not concurrently (fix the interleaved log problem)
  each stage wrapped in StageResult with: name, duration, exit_status, warnings
  __repr__ produces the Phase 5A visual standard terminal output
```

**MIT license header required on every file:**
```python
# MIT License
# Copyright (c) 2026 [project owner]
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction...
```

---

## SECTION 2 — CRITICAL BUGS (IN PRIORITY ORDER)

**BUG ZERO [NEW — CONFIRMED LIVE] — Evaluate.py produces F1=0 on valid model**
Priority: BLOCKING. No other work matters until this is resolved.
Evidence: Pretrained CORD model scores 0.1533 on CORD test (should be >0.90). Experiment 2 scores 0.0 on SROIE test.
Investigation path:
  1. Print 3 raw model outputs (token sequences) before any parsing. Are they well-formed?
  2. Print 3 ground-truth labels as they exist in the dataset. Do field names match SROIE key files?
  3. Print the result of token2json() on a raw output. Does it return the expected dict structure?
  4. Print the F1 computation inputs (pred dict, label dict) for one sample. Is the key matching correct?
Fix must include: a self-test method on DonutEvaluator that runs on 1 sample and asserts F1 > 0 before evaluating the full set.

**BUG CRITICAL [CONFIRMED LIVE] — decoder.lm_head.weight missing on load**
Evidence: `There were missing keys in the checkpoint model loaded: ['decoder.lm_head.weight']`
The Hugging Face VisionEncoderDecoderModel uses tied weights between the encoder's projection and the decoder LM head.
When saving with `save_pretrained()`, the tied weight may be excluded from the state dict.
When loading, the missing key must be explicitly handled — either by tying weights after load or by saving the full (untied) state dict.
Fix: after `model.from_pretrained(path)`, explicitly call `model.decoder.lm_head.weight = model.decoder.model.decoder.embed_tokens.weight` IF the checkpoint was saved with tied weights. Add an assertion: `assert model.decoder.lm_head.weight is not None`.

**BUG CONFIRMED [PRIOR] A — SROIE key file format mismatch**
Patched in prior iteration. Verify the fix: `_load_key_file()` must try `.txt` first (4-line format: company, date, address, total), then `.json`. Add a unit assertion: load one known key file and assert all 4 field keys are present and non-empty.

**BUG CONFIRMED [PRIOR] B + 2 — SROIE-NER loader broken**
`load_dataset("darentang/sroie")` no longer works. Parquet fallback implemented previously.
Verify the parquet URL is still live. Verify column names in the parquet file match what the code expects (`words`, `ner_tags`, `image_path`). If the parquet API is broken, this loader must raise `DatasetLoadError` loudly — it must never silently return [].

**BUG CONFIRMED [PRIOR] C — SROIE_DATA_DIR env var not propagated**
Module-level constant evaluated before env var is set. Fix verified: `_get_sroie_dir()` function must be called at use time, not at import time. Add an assertion in every loader that reads SROIE data: `assert Path(_get_sroie_dir()).exists(), f"SROIE dir not found: {_get_sroie_dir()}"`.

**BUG CONFIRMED [PRIOR] 1 — WildReceipt silent empty return**
Direct OpenMMLab tar download implemented previously. Verify: the tar URL `https://download.openmmlab.com/mmocr/data/wildreceipt.tar` is live. Verify the extracted structure matches what the loader expects. If download fails, this must raise `DatasetLoadError`, not return [].

**BUG CONFIRMED [PRIOR] 3 — --force flag ignores cached results**
`results/*.json` must be deleted when `--force` is passed. Verify this is implemented. Add a test: run with `--force`, confirm JSON files are regenerated with new timestamps.

**BUG CONFIRMED [PRIOR] 5 — token2json crash on malformed output**
Wrapped in try/except. This is necessary but not sufficient. The except branch must: log the raw token string that failed to parse, increment a `parse_failure_count` counter, and return `{}` with all FIELDS set to empty string (not omit keys entirely). If parse_failure_count > 50% of samples, raise an exception — the model is broken, not the data.

**BUG CONFIRMED [PRIOR] 6 — Epoch count mismatch**
`num_train_epochs` must come from `ExperimentConfig` exclusively. There must be zero hardcoded epoch values in train.py. The value in run_experiments.py's TRAIN_CONFIG is the single source. Add an assertion: `assert config.epochs == training_args.num_train_epochs`.

**BUG CONFIRMED [PRIOR] 7 — CORD date field hardcoded to empty**
`_cord_remap()` must extract `date_info` from `gt_parse`. Add a test: load 5 CORD samples, assert that `date` field is non-empty in at least 3 of them.

**BUG CONFIRMED [PRIOR] 9 — Silent 0-sample degradation**
`get_combined_dataset()` prints a warning. This is not enough. If any dataset that an experiment explicitly declares as a required source returns 0 samples, the experiment must FAIL with a `DatasetLoadError`, not silently train on a subset. The warning is appropriate only for optional auxiliary datasets.

**BUG SUSPECTED — Stale cache without schema validation**
`.downloaded` marker files check presence, not correctness. `CORDLoader.validate_cache()` and `InvoicesDonutLoader.validate_cache()` must check: (1) expected file count, (2) a sample JSON schema spot-check on the first record, (3) parquet column names match expected. If validation fails, delete the cache and re-download.

**BUG SUSPECTED — CORD image filename collision on multi-load**
`load_cord()` uses `idx = len(samples)` for image filenames. This is not idempotent. Use a content hash (`hashlib.md5(image.tobytes()).hexdigest()[:8]`) as the filename suffix instead of a counter.

**BUG SUSPECTED — run_all.py docstring vs. actual count mismatch**
Docstring says 7 experiments. `inject_results.py` EXP_NAMES has keys 1–8. Audit this. The correct number is whatever `run_experiments.py` actually defines. The docstring must match. If there are 8 experiments, the paper must describe 8.

**BUG SUSPECTED — inject_results.py hardcoded leaderboard scores**
`DONUT_ZEROSHOT_F1 = 0.8411` and LEADERBOARD scores (H&H Lab = 0.9567, CLOVA OCR = 0.9373) must be verified against the source papers. If they are wrong, the paper contains fabricated comparisons. Add inline citations as comments with DOI or arXiv links.

---

## SECTION 3 — LINE-COUNT TARGETS PER FILE (MINIMUM 3,800 TOTAL)

These are minimum substantive line changes. Blank lines, comments-only, and import reorders do not count.

```
FILE                   MINIMUM LINES    RATIONALE
─────────────────────────────────────────────────────────────────────────
run_all.py             600              PipelineOrchestrator class, Phase 5A
                                        visual output, sequential stage
                                        execution, StageResult wrapper,
                                        --force cache deletion, MIT header

run_experiments.py     400              ExperimentConfig dataclass, single
                                        source of truth for all training
                                        hyperparams, remove all duplicate
                                        config values, MIT header

dataset_loaders.py     900              BaseDatasetLoader ABC + 5 concrete
                                        subclasses, schema-validating cache,
                                        DatasetLoadError exception class,
                                        CORD hash-based filenames, loud
                                        zero-sample guard, MIT header

train.py               600              DonutTrainer class, zero-sample guard
                                        in __init__, tied-weight fix for
                                        decoder.lm_head, ExperimentConfig
                                        integration, remove duplicate
                                        hyperparams, MIT header

evaluate.py            600              DonutEvaluator class, self-test on 1
                                        sample before full eval, parse_failure
                                        counter with hard threshold, correct
                                        NED direction documentation, F1 empty-
                                        pred behavior documented, MIT header

inject_results.py      300              PaperInjector class, unresolved-VAR
                                        exception, leaderboard inline citations,
                                        dynamic sample counts only, MIT header

paper.tex              200              siunitx + xcolor packages, all \VAR{}
                                        resolved verified, direction indicators
                                        on all metric columns, sample size in
                                        captions, zero-shot dagger footnote,
                                        8-experiment table if 8 experiments

requirements.txt       100              Version-pinned all deps, pyarrow pandas
                                        editdistance tqdm confirmed present,
                                        comment block explaining each pin reason
─────────────────────────────────────────────────────────────────────────
TOTAL                  3,700 (min)      + incidental fixes = 3,800+ total
```

---

## SECTION 4 — IMPLEMENTATION CONSTRAINTS

**Concurrency:** Experiments run SEQUENTIALLY. The interleaved log streams and eval runtime spikes in the live run are a direct consequence of concurrent execution. Sequential execution is slower but produces trustworthy logs and reproducible GPU memory behavior.

**Failing loudly:** Every function that can return an empty result must raise an exception instead of returning []. The only acceptable silent return is `{}` from `_parse_prediction()`, and even that must log the failure. Every FATAL condition must print to stderr with the text `FATAL:` at the start of the line so it is grep-able.

**No global state:** Module-level constants that can change at runtime (e.g., SROIE_DATA_DIR) must be removed. All paths must be resolved through instance variables or method parameters at call time.

**No hardcoded experiment results:** Any number that appears in paper.tex must be sourced from results JSON files. Any number that appears in inject_results.py that is not from a results JSON must have an inline citation comment with a verifiable source.

**Config single-source:** `ExperimentConfig` in run_experiments.py is the one and only place where training hyperparameters live. train.py reads from the config object. No hardcoded epochs, lr, or batch size in train.py.

**Cache invalidation:** All dataset caches must be invalidated and re-downloaded if: (a) `--force` is passed, (b) `validate_cache()` returns False, or (c) the code version hash stored in the cache metadata differs from the current version.

**Evaluation self-test:** `DonutEvaluator.evaluate()` must run a single-sample self-test before the full evaluation loop. The self-test loads one image, runs inference, parses the output, and asserts that the result is a dict with at least one non-empty field. If the self-test fails, the method raises immediately with a diagnostic message that includes the raw token output. This would have caught both the F1=0.1533 anomaly and the Experiment 2 F1=0 failure at the earliest possible moment.

---

## SECTION 5 — PRE-SHIP VERIFICATION CHECKLIST (MUST BE COMPLETED)

Every box must be explicitly checked in the PR description. An unchecked box blocks the PR.

```
FILE                  CRITERION                                           STATUS
────────────────────────────────────────────────────────────────────────────────
run_all.py            No concurrent experiment execution                    □
run_all.py            Phase 5A visual output standard implemented           □
run_all.py            --force deletes all cached results                    □
run_all.py            Substantive change (≥600 lines)                       □

run_experiments.py    ExperimentConfig is single source for all hyperparams □
run_experiments.py    No duplicate config values in train.py               □
run_experiments.py    Substantive change (≥400 lines)                       □

dataset_loaders.py    All loaders raise DatasetLoadError, never return []  □
dataset_loaders.py    Cache validation checks schema, not just presence     □
dataset_loaders.py    CORD filename collision fixed                         □
dataset_loaders.py    Substantive change (≥900 lines)                       □

train.py              decoder.lm_head.weight missing-key fix verified       □
train.py              Zero-sample raises ValueError in __init__              □
train.py              No hardcoded hyperparams (all from ExperimentConfig)  □
train.py              Substantive change (≥600 lines)                       □

evaluate.py           Single-sample self-test before full evaluation loop   □
evaluate.py           F1 = 0.1533 anomaly diagnosed and root cause fixed    □
evaluate.py           parse_failure_count threshold implemented             □
evaluate.py           NED direction documented and verified correct         □
evaluate.py           Substantive change (≥600 lines)                       □

inject_results.py     Unresolved \VAR{} raises exception                   □
inject_results.py     Leaderboard scores have inline citation comments      □
inject_results.py     No hardcoded experiment F1/NED values                 □
inject_results.py     Substantive change (≥300 lines)                       □

paper.tex             All \VAR{} resolved (zero remaining)                  □
paper.tex             F1 column has ↑ indicator, NED has ↓                 □
paper.tex             siunitx for decimal alignment present                 □
paper.tex             Zero-shot baseline has † and footnote                 □
paper.tex             Substantive change (≥200 lines)                       □

requirements.txt      All imports have corresponding pinned deps            □
requirements.txt      MIT license comment block present                     □
requirements.txt      Substantive change (≥100 lines)                       □

TOTAL LINE DELTA      ≥3,800 substantive lines across all 8 files           □
SELF-TEST PASSES      DonutEvaluator.evaluate() self-test passes on 1 sample □
F1 SANITY             Pretrained CORD model on CORD test scores >0.80       □
F1 SANITY             Experiment 2 model on SROIE test scores >0.0          □
```

---

## SECTION 6 — VERIFIED RESOLVED CHECKLIST (CONFIRM IN PR)

```
BUG   ORIGINAL SYMPTOM                          CONFIRMED FIX LOCATION
───────────────────────────────────────────────────────────────────────────
A     SROIE key file .json vs .txt format       dataset_loaders.py _load_key_file()
B     SROIE-NER datasets script removed         dataset_loaders.py SROIENERLoader
C     SROIE_DATA_DIR env var not propagated     dataset_loaders.py _get_sroie_dir()
D     Test split key extension mismatch         run_all.py stage_install()
E     train.py duplicate .json-only bug         train.py SROIEDataset or DonutTrainer
1     WildReceipt silent empty return           dataset_loaders.py WildReceiptLoader
2     SROIE-NER download method broken          dataset_loaders.py SROIENERLoader
3     --force flag ignores cached results       run_all.py PipelineOrchestrator
5     token2json crash on malformed output      evaluate.py DonutEvaluator._parse_prediction()
6     Epoch count mismatch train.py vs config   ExperimentConfig + assertion
7     CORD date field hardcoded empty           dataset_loaders.py CORDLoader._cord_remap()
9     Silent 0-sample degradation               BaseDatasetLoader.load() zero-guard
NEW   F1 = 0.1533 on pretrained CORD model      evaluate.py self-test + root cause fix
NEW   F1 = 0 on Experiment 2                    train.py decoder.lm_head fix
NEW   Interleaved log / concurrent experiments  run_all.py sequential execution
```

---

## SECTION 7 — SHIPPING STANDARD

The shipping standard is not "does it run."
The shipping standard is:

- If a number appears in the final output, it is real.
- If a label appears in the final output, it matches what was computed.
- If a file is loaded, its content matches what the code expects.
- If a file is in the manifest, it contains a substantive change.
- If an experiment reports F1 = 0, the pipeline refuses to write that result to paper.tex and raises an error.
- If the pretrained baseline cannot score above 0.50 on its own benchmark, the pipeline refuses to proceed and prints a FATAL diagnostic.
- No exceptions.

---

*Evidence base: live training log 2026-02-23 12:20–ongoing (vast.ai, RTX 4090, CUDA 12.8) + FORENSIC_AUDIT_PROMPT_V2_FILLED.md v2.0*
*8 files in scope. 3,800 minimum substantive lines. 16 bugs confirmed or suspected. 3 new bugs confirmed live.*
