# FORENSIC SYSTEM AUDIT & FULL PIPELINE REBUILD — AGENT PROMPT
> **Classification:** Production-Grade Code Integrity Protocol  
> **Scope:** Full-pipeline rebuild after total failure + dual-architecture comparison + IEEE-quality paper generation  
> **Standard:** Every output must be real, labeled, and reproducible. No exceptions.  
> **Target venue quality:** IEEE Transactions on Image Processing / International Journal of Computer Vision (IJCV)

---

```
╔══════════════════════════════════════════════════════════════════════════════╗
║     FULL PIPELINE REBUILD + DUAL-ARCHITECTURE COMPARISON                    ║
║     Post-merge PR — All prior changes are on main                           ║
║     Total pipeline failure on last run — 0/8 experiments completed           ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## CONTEXT: WHAT HAPPENED

A previous PR was merged into `main` on branch `claude/review-project-files-cRckT`. That PR attempted to fix critical bugs (lm_head weight tying, train-on-test data leakage, early_stopping deprecation) and add a parallel TrOCR+YOLO pipeline. **However, the subsequent run on a vast.ai RTX 4090 instance failed completely.** All 8 DONUT experiments crashed at `DonutProcessor.from_pretrained()` because `protobuf` was not in `requirements.txt`. The XLMRoberta tokenizer (used internally by DONUT's processor) requires `protobuf>=3.20.0` at runtime. The TrOCR+YOLO pipeline was never reached.

**This is a NEW PR.** The previous PR is already merged. You are creating a fresh branch from current `main` and opening a new PR with all necessary fixes and the complete dual-architecture pipeline.

### Terminal failure summary (from the last run):
- **Environment:** Python 3.12.12, torch 2.10.0+cu126, transformers 4.57.6, RTX 4090 (24GB), 62GB RAM
- **Stage 0 (SROIE Install):** ✓ OK — 500 train / 63 val / 63 test (the shutil.move fix from prior PR worked)
- **Stage 1 (Dataset Download):** ✓ OK (with 1 warning) — wildreceipt=886, cord=630, invoices_donut=332 train samples loaded
- **Stage 1 model pre-download WARNING:** `protobuf` missing — XLMRobertaConverter import failed, but stage continued because the warning was non-fatal
- **Stage 1.5 (Pretrained Baseline):** ✗ FAIL — `ImportError: XLMRobertaConverter requires the protobuf library`
- **Stage 2 (Experiments 1–8):** ✗ ALL 8 CRASHED — Same `protobuf` ImportError at `DonutProcessor.from_pretrained()`
- **Stage 3 (Paper Generation):** ✗ FAIL — 105 unresolved `\VAR{}` placeholders (no experiment data to inject)
- **Total wall time:** 5.4 min, exit code 1
- **Zero experiments trained. Zero models produced. Zero metrics computed.**

### Prior bugs that were fixed in the merged PR (verify these are still correct on main):
1. **lm_head weight tying** — `model.decoder.config.tie_word_embeddings = False` after `resize_token_embeddings()`; destructive `model.tie_weights()` removed from `DonutTrainer.save()`
2. **Train-on-test data leakage** — `shutil.copy2()` changed to `shutil.move()` in `stage_install()` so val/test images are removed from `img/`
3. **early_stopping deprecation** — Removed `early_stopping=True` from `model.generate()` calls (invalid with `num_beams=1`)
4. **Paper accuracy** — Corrected "zero-shot CORD-to-SROIE transfer" claim; published 84.11% is SROIE-fine-tuned
5. **Conditional _retie_decoder_head** — Skips re-tie for new checkpoints where `tie_word_embeddings=False`

---

## ═══ MISSION BRIEFING ═══

You are operating as a cross-disciplinary forensic strike team. Your job is to:

1. **Fix the immediate blocker** (`protobuf` missing) and any other dependency issues
2. **Audit and fix every file** so the full pipeline runs end-to-end without failure
3. **Build a complete dual-architecture comparison** (DONUT vs TrOCR+YOLO) with systematic experimental design
4. **Generate an IEEE-quality LaTeX paper** with all metrics dynamically injected
5. **Ensure the entire pipeline is reproducible** from `pip install -r requirements.txt && python run_all.py`

Your standard is not "does it run without crashing."  
Your standard is: **is every output correct, reproducible, and trustworthy? Would this survive peer review at IJCV?**

---

## ═══ PHASE 1 — KNOWN ISSUES (CARRY FORWARD — DO NOT ASSUME RESOLVED) ═══

### BLOCKING: Missing dependency
```
protobuf>=3.20.0    # REQUIRED by XLMRobertaConverter (Donut tokenizer)
                    # This single missing dependency caused 100% pipeline failure
                    # Add to requirements.txt IMMEDIATELY
```

### Critical issues from prior audit (verify on current main):

```
Issue                          → Symptom                              → Root cause                           → Prior fix status
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
protobuf missing               → ImportError on ALL DonutProcessor    → Not in requirements.txt              → UNFIXED (blocker)
                                 .from_pretrained() calls                                                      
lm_head weight tying           → F1=0 on all experiments              → tie_weights() overwrites learned     → Fixed in merged PR
                                                                        lm_head; must set                      (VERIFY still correct)
                                                                        tie_word_embeddings=False               
train-on-test data leakage     → 626 train samples instead of 500    → shutil.copy2 didn't remove           → Fixed in merged PR
                                                                        val/test from img/                     (VERIFY still correct)
early_stopping deprecation     → Thousands of warnings per eval       → early_stopping=True invalid           → Fixed in merged PR
                                                                        with num_beams=1                       
stale cache miss               → Changed warmup_steps not detected    → TRAIN_CONFIG doesn't include          → UNFIXED
                                                                        warmup_steps, weight_decay, etc.       
constants duplicated 5x        → Silent drift if any file updated     → FIELDS, IMAGE_EXTS, etc. defined     → UNFIXED
                                                                        independently in 5+ files              
no GPU cleanup between exps    → Potential OOM on smaller GPUs        → No del model; torch.cuda.             → UNFIXED
                                                                        empty_cache() between experiments       
no HF download timeout         → Pipeline hangs indefinitely          → load_dataset() has no timeout         → UNFIXED
paper prose assumes outcomes   → Wrong analysis if results differ     → Hardcoded "jump from Exp 1 to 4"     → UNFIXED
missing references.bib         → LaTeX warning on compile             → Dangling \bibliography{references}    → UNFIXED
```

### Suspected but unconfirmed issues to investigate:
- The pretrained CORD baseline previously scored F1=0.1589 (should be ~0.84) — is the CORD-to-SROIE field remapping broken?
- `company_ned = 1.0` across all prior experiments — is the company field extraction systematically failing?
- WildReceipt loaded 1,267 samples but README claims ~1,740 — are samples being silently dropped during schema remapping?
- Invoices loaded 475 but claimed ~800 — same question
- Log files (log1.txt, log2.txt, log3.txt) and stale experiment JSONs committed to repo — need cleanup
- `SROIENERLoader` defined in dataset_loaders.py but never used — dead code
- `hf_token.txt` is in .gitignore but tracked in git history — security risk (DO NOT touch the token itself, just note it)

---

## ═══ PHASE 2 — THE EXPERIMENTAL DESIGN (WHAT THE PIPELINE MUST DO) ═══

The project implements a **systematic comparison of end-to-end vs. pipeline architectures for receipt information extraction**, benchmarked against the ICDAR 2019 SROIE competition. The design has 7 stages:

### Stage 1: Dataset Acquisition & Normalization
Take receipt/invoice images from 4 sources. Normalize ALL to the SROIE schema `{company, date, address, total}`:

| Dataset | Source | Raw fields | Normalization needed |
|---------|--------|-----------|---------------------|
| **SROIE** | GitHub clone (zzzDavid/ICDAR-2019-SROIE) | 4 fields: company, date, address, total | None (native schema) |
| **WildReceipt** | OpenMMLab tar download | 25 KIE categories | Remap to 4 SROIE fields |
| **CORD v2** | HuggingFace (naver-clova-ix/cord-v2) | 30+ fields (store_info, total, date, etc.) | Remap to 4 SROIE fields |
| **Invoices-DONUT** | HuggingFace (katanaml-org/invoices-donut-data-v1) | 7+ invoice fields | Remap to 4 SROIE fields |

### Stage 2: Data Splitting & Isolation
- SROIE: 80/10/10 split → 500 train / 63 val / 63 test (val+test MOVED out of train dir, not copied)
- Auxiliary datasets: 70/15/15 split → only train+val used (test discarded to prevent contamination)
- **The 63 SROIE test images are the ONLY test set for ALL experiments** (both DONUT and TrOCR+YOLO)
- This ensures apples-to-apples comparison across all experiments

### Stage 3: DONUT Experiments (Architecture A — End-to-End)
Fine-tune DONUT (`naver-clova-ix/donut-base`) on 8 dataset combinations, evaluate each on the same 63 SROIE test images:

| Exp | Training data | Expected train samples |
|-----|--------------|----------------------|
| 1 | SROIE only (baseline) | 500 |
| 2 | SROIE + WildReceipt | 500 + 886 = 1,386 |
| 3 | SROIE + Invoices-DONUT | 500 + 332 = 832 |
| 4 | SROIE + CORD | 500 + 630 = 1,130 |
| 5 | SROIE + WildReceipt + Invoices | 500 + 886 + 332 = 1,718 |
| 6 | SROIE + WildReceipt + CORD | 500 + 886 + 630 = 2,016 |
| 7 | SROIE + Invoices + CORD | 500 + 332 + 630 = 1,462 |
| 8 | SROIE + All three | 500 + 886 + 630 + 332 = 2,348 |

Additionally: evaluate the pretrained `naver-clova-ix/donut-base-finetuned-cord-v2` as a zero-shot baseline (CORD→SROIE transfer, no fine-tuning).

### Stage 4: TrOCR+YOLO Experiments (Architecture B — Pipeline)
Two-stage pipeline: YOLOv8 detects text regions → TrOCR reads text from crops → heuristic post-processing assigns fields.

Run the SAME 8 dataset combinations as DONUT, evaluate on the SAME 63 SROIE test images. This gives a matched experimental design for fair comparison.

| Component | Model | Parameters |
|-----------|-------|-----------|
| Detection | YOLOv8n | ~3.2M params |
| Recognition | TrOCR-base | ~334M params |
| Field assignment | Rule-based heuristics | N/A |
| **Total** | | **~337M+ params, 2 fine-tuning stages** |

vs. DONUT: ~200M params, 1 fine-tuning stage.

### Stage 5: Cross-Architecture Comparison
Compare DONUT vs TrOCR+YOLO on:
- **Performance:** Global F1, per-field F1, precision, recall, NED, exact match
- **Complexity:** Parameter count, fine-tuning stages (1 vs 2), training time, inference time
- **Error propagation:** End-to-end (DONUT: none) vs cascading (TrOCR+YOLO: detection error → OCR error → field assignment error)
- **Robustness:** How each degrades with out-of-domain data, noisy images, non-English text

### Stage 6: Comparison with SROIE Contest Results
Compare best results from both architectures against published leaderboard:

| Method | F1 (%) | Year |
|--------|--------|------|
| LayoutLMv3 | 96.33 | 2022 |
| PICK | 96.12 | 2021 |
| H&H Lab (ICDAR'19 1st) | 95.67 | 2019 |
| BROS | 95.48 | 2022 |
| LayoutLMv2 | 94.95 | 2021 |
| CLOVA OCR (ICDAR'19 2nd) | 93.73 | 2019 |
| ICDAR'19 3rd | 91.98 | 2019 |
| DONUT (SROIE fine-tuned, Kim et al.) | 84.11 | 2022 |

**IMPORTANT:** Our test split (63 images from the 626 annotated training set) is NOT the official SROIE test set (347 images with no public ground truth). The paper MUST state this clearly with a footnote. Leaderboard numbers are NOT directly comparable — they are provided for contextual reference only.

### Stage 7: LaTeX Paper Generation
Generate a complete IEEE-format paper with ALL metrics dynamically injected from experiment JSON files. No hardcoded numbers in the paper template. The paper must include:
- Dataset statistics table (with actual loaded counts, not README claims)
- Per-experiment results table (DONUT, 8 experiments)
- Per-experiment results table (TrOCR+YOLO, 8 experiments)
- Per-field breakdown table (both architectures)
- Cross-architecture comparison table (performance + complexity)
- Convergence plots (training loss curves from logged training history)
- Leaderboard comparison table (with proper caveats about non-comparable test split)
- Statistical analysis: confidence intervals, significance tests where appropriate

---

## ═══ PHASE 3 — FILE MANIFEST ═══

These are the core files. ALL must be audited, fixed, and substantively changed:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  FILE MANIFEST                                                            │
├────┬─────────────────────────────┬───────────────────────────────────────┤
│ 01 │ requirements.txt            │ Add protobuf>=3.20.0, verify ALL deps │
│ 02 │ run_all.py                  │ Pipeline orchestrator — all stages     │
│ 03 │ dataset_loaders.py          │ Multi-dataset download & normalize     │
│ 04 │ run_experiments.py          │ DONUT experiment configs & training    │
│ 05 │ train.py                    │ DONUT fine-tuning trainer              │
│ 06 │ evaluate.py                 │ SROIE Task-3 evaluation metrics        │
│ 07 │ inject_results.py           │ LaTeX paper generation & var injection │
│ 08 │ paper.tex                   │ IEEE paper template with \VAR{} slots  │
│ 09 │ 01_dataset_preparation.py   │ YOLO bbox + TrOCR crop generation      │
│ 10 │ 02_train_donut.py           │ Standalone DONUT training (reference)  │
│ 11 │ 03_train_trocr_yolo.py      │ YOLOv8 + TrOCR two-stage training     │
│ 12 │ 04_evaluate.py              │ Unified eval for both architectures    │
│ 13 │ 05_compare_results.py       │ Cross-architecture comparison & plots  │
└────┴─────────────────────────────┴───────────────────────────────────────┘
```

**Agent instruction:** You must open, read, and produce at least one substantive change in each of these files. A "substantive change" means a fix, refactor, assertion, guard, or structural improvement that meaningfully affects correctness, robustness, or clarity. Renaming a variable or adjusting whitespace does not qualify.

---

## ═══ PHASE 4 — REQUIREMENTS.TXT FIX (DO THIS FIRST) ═══

The immediate blocker. Add this line to `requirements.txt`:

```
protobuf>=3.20.0          # required by XLMRoberta tokenizer (Donut)
```

Then audit ALL other dependencies against what was actually installed in the terminal log:
- `transformers` pinned `<5.0.0` but 4.57.6 was installed (OK, within range)
- `torch>=2.0.0` — 2.10.0 installed (OK)
- `datasets<5.0.0,>=2.14.0` — 4.6.0 installed (OK)
- Verify `ultralytics` (for YOLOv8) is present for the TrOCR+YOLO pipeline
- Verify `timm` is present (TrOCR may need it)
- Check for any other transitive dependencies that could cause runtime ImportError

---

## ═══ PHASE 5 — CONSTANTS DEDUPLICATION ═══

Create a `constants.py` (or equivalent shared module) to eliminate the 5x duplication:

```python
# These are currently defined independently in 5+ files:
FIELDS = ["company", "date", "address", "total"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MAX_LENGTH = 512
BASE_MODEL = "naver-clova-ix/donut-base"
SEED = 42
NEW_TOKENS = ["<s_sroie>", "</s_sroie>", "<s_company>", "<s_date>", "<s_address>", "<s_total>",
              "</s_company>", "</s_date>", "</s_address>", "</s_total>"]
```

All files must import from this single source. If any constant changes, it changes once.

---

## ═══ PHASE 6 — CACHE VALIDATION FIX ═══

In `run_experiments.py`, `TRAIN_CONFIG` only includes `max_epochs`, `learning_rate`, `per_device_train_batch_size`, `early_stopping_patience`, and `base_model`. But `ExperimentConfig` also has `warmup_steps`, `weight_decay`, `max_length`, and `seed`. The cache validation at `cached.get("config") != TRAIN_CONFIG` will NOT detect stale results if `warmup_steps` changes from 100 to 200.

**Fix:** Include ALL hyperparameters in the cache validation dict. Or better: hash the entire `ExperimentConfig` and store it with the result.

---

## ═══ PHASE 7 — GPU MEMORY MANAGEMENT ═══

Between experiments, add explicit cleanup:
```python
del model, trainer
torch.cuda.empty_cache()
gc.collect()
```

The RTX 4090 has 24GB — sufficient for DONUT (~200M params) but TrOCR+YOLO combined could push limits. Explicit cleanup prevents OOM when running 8 DONUT experiments followed by 8 TrOCR+YOLO experiments.

---

## ═══ PHASE 8 — TROCR+YOLO PIPELINE INTEGRATION ═══

The reference files (01-05) were added in the merged PR but may need further adaptation. Key requirements:

1. **01_dataset_preparation.py** must generate YOLO-format bbox labels and TrOCR line crops FROM the existing SROIE split (not re-downloading or re-splitting)
2. **03_train_trocr_yolo.py** must run the SAME 8 dataset combinations as DONUT experiments
3. **04_evaluate.py** must evaluate BOTH architectures on the SAME 63 SROIE test images using the SAME metrics (F1, precision, recall, NED, exact match)
4. **05_compare_results.py** must produce comparison tables and plots suitable for LaTeX injection
5. All TrOCR+YOLO results must be saved in the same JSON format as DONUT results for uniform paper generation

### Error cascading analysis (unique to TrOCR+YOLO):
The paper must analyze how detection errors propagate through the pipeline:
- If YOLO misses a text region → that field gets no prediction → recall drops
- If YOLO crops the wrong region → TrOCR reads garbage → precision drops
- If TrOCR misreads text → field assignment heuristic may assign to wrong field → both precision and recall affected

This cascading error structure is a key scientific contribution — it demonstrates why end-to-end models (DONUT) are architecturally advantageous even if individual components of the pipeline approach have higher standalone accuracy.

---

## ═══ PHASE 9 — PAPER QUALITY REQUIREMENTS (IEEE/IJCV LEVEL) ═══

The paper must reach the quality bar of IEEE Transactions on Image Processing. This means:

### Scientific rigor:
- **Hypothesis clearly stated:** "Cross-domain data augmentation improves receipt IE, and end-to-end architectures (DONUT) are more robust than pipeline approaches (TrOCR+YOLO) despite lower individual component accuracy"
- **Experimental design is controlled:** Same test set, same metrics, same hardware, same random seed
- **Statistical validity:** Report mean ± std where possible; note that single-seed results limit statistical claims
- **Limitations section:** Non-comparable test split, single GPU, limited hyperparameter search, English-centric evaluation
- **Ablation-style analysis:** The 8 experiments ARE the ablation — each adds one dataset, measuring marginal contribution

### Paper structure:
1. Abstract (dynamically generated from best results)
2. Introduction (motivation: receipt IE is a $B industry; gap: no systematic comparison of end-to-end vs pipeline on cross-domain augmentation)
3. Related Work (DONUT, TrOCR, YOLO, SROIE competition history, LayoutLM family)
4. Methodology
   - 4.1 Dataset acquisition and normalization
   - 4.2 DONUT architecture and fine-tuning
   - 4.3 TrOCR+YOLO architecture and two-stage training
   - 4.4 Evaluation protocol (SROIE Task-3 metrics)
5. Experiments
   - 5.1 Dataset statistics (actual loaded counts with provenance)
   - 5.2 DONUT results (8 experiments + zero-shot baseline)
   - 5.3 TrOCR+YOLO results (8 experiments)
   - 5.4 Cross-architecture comparison
   - 5.5 Comparison with published results (with caveats)
   - 5.6 Error analysis and failure cases
   - 5.7 Complexity and cost analysis
6. Discussion
7. Conclusion and Future Work
8. References (proper .bib file, not inline thebibliography)

### LaTeX requirements:
- All numbers from `\VAR{}` placeholders dynamically injected — ZERO hardcoded metrics
- All tables use `booktabs` (no vertical lines, proper `\toprule`/`\midrule`/`\bottomrule`)
- Direction indicators on every metric column: F1↑, NED↓, Precision↑, Recall↑
- Sample sizes in table captions
- Zero-shot baselines marked with `†` and footnoted
- Convergence plots from actual training logs (not synthetic data)
- Best result in each column **bolded**

### Data that must be collected during training (meticulous logging):
For EVERY experiment (both DONUT and TrOCR+YOLO):
- Training loss per epoch
- Validation loss per epoch
- Learning rate schedule
- Wall-clock training time
- Wall-clock inference time (per image and total)
- GPU memory peak usage
- Number of training steps
- Best epoch (early stopping trigger point)
- Per-field F1, precision, recall, NED
- Global F1, precision, recall, exact match
- Parse failure count and rate
- Sample predictions (for qualitative analysis / error examples)

All of this goes into the experiment JSON files and is available for paper generation.

---

## ═══ PHASE 10 — END-TO-END PIPELINE INTEGRITY VERIFICATION ═══

Trace the complete data flow and verify at every step:

### Data leakage checks:
- [ ] SROIE val/test images are MOVED (not copied) from `img/` — verify `img/` has exactly 500 files after split
- [ ] Auxiliary datasets use only their train+val splits (test discarded)
- [ ] No SROIE test images appear in any training set (for both DONUT and TrOCR+YOLO)
- [ ] The same 63 test images are used for ALL evaluations

### Sample count integrity:
- [ ] Report counts from actual loaded data, not config values
- [ ] Log and verify: SROIE train=500, val=63, test=63
- [ ] Log auxiliary counts after schema remapping (some samples may be dropped if fields are empty)
- [ ] Flag if any dataset loads <90% of expected samples

### Metric integrity:
- [ ] F1 computation matches SROIE Task-3 specification (entity-level, not token-level)
- [ ] NED uses consistent denominator (document which convention: max(len(pred), len(gt)) or len(gt))
- [ ] Exact match counts only full 4-field matches
- [ ] Zero-shot baseline uses CORD→SROIE field remapping (verify the mapping is correct)

### Reproducibility:
- [ ] Fixed seed (42) for all random operations
- [ ] Deterministic data splits
- [ ] Model checkpoints saved with full config
- [ ] Results JSON includes all hyperparameters, dataset versions, library versions

---

## ═══ PHASE 11 — TERMINAL OUTPUT STANDARDS ═══

All terminal output must use structured formatting:

```
╔══════════════════════════════════════════════════════════════╗
║  STAGE [N] — [STAGE NAME]                                    ║
╚══════════════════════════════════════════════════════════════╝

  ✓  [Success item]         : [value or confirmation]
  ✗  [Failure item]         : [diagnosis]
  ⚠  [Warning item]         : [detail + affected experiment IDs]

  ┌─────────────────────────────────────────────┐
  │  METRIC TABLE                               │
  ├─────────────────┬───────────┬───────────────┤
  │  Field          │  F1 ↑     │  NED ↓        │
  ├─────────────────┼───────────┼───────────────┤
  │  company        │  0.8800   │  0.0370       │
  │  date           │  0.9091   │  0.0356       │
  │  address        │  0.4286   │  0.1012       │
  │  total          │  0.8643   │  0.0647       │
  ├─────────────────┼───────────┼───────────────┤
  │  GLOBAL         │  0.7718   │  —            │
  └─────────────────┴───────────┴───────────────┘

  Finished in [X.X] min  |  Exit: [SUCCESS / FAILURE]
```

Rules:
- Zero-sample datasets MUST raise a hard exception, not a warning
- Every WARNING must state which experiment IDs are affected
- Timing per-stage AND per-experiment
- Color codes degrade gracefully to non-TTY (pipe to file must be readable)

---

## ═══ PHASE 12 — PRE-SHIP VERIFICATION CHECKLIST ═══

Before the PR is submitted, verify EVERY box:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PRE-SHIP VERIFICATION CHECKLIST                                          │
├────┬─────────────────────────────┬──────────────────────────────────┬────┤
│ ## │ File                        │ Verification criterion            │ ✓  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 01 │ requirements.txt            │ protobuf>=3.20.0 present          │ □  │
│ 01 │ requirements.txt            │ ALL deps verified against runtime │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 02 │ run_all.py                  │ All stages execute without crash  │ □  │
│ 02 │ run_all.py                  │ TrOCR+YOLO stage integrated       │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 03 │ dataset_loaders.py          │ No silent fallbacks mask failure  │ □  │
│ 03 │ dataset_loaders.py          │ Schema remapping verified correct │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 04 │ run_experiments.py          │ Cache validates ALL hyperparams   │ □  │
│ 04 │ run_experiments.py          │ Constants imported from shared    │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 05 │ train.py                    │ tie_word_embeddings=False after   │ □  │
│    │                             │ resize_token_embeddings()         │    │
│ 05 │ train.py                    │ No destructive tie_weights()      │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 06 │ evaluate.py                 │ No early_stopping=True            │ □  │
│ 06 │ evaluate.py                 │ Metric labels match computation   │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 07 │ inject_results.py           │ All \VAR{} placeholders resolved  │ □  │
│ 07 │ inject_results.py           │ TrOCR+YOLO results injected      │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 08 │ paper.tex                   │ No hardcoded metrics              │ □  │
│ 08 │ paper.tex                   │ Proper .bib references            │ □  │
│ 08 │ paper.tex                   │ Non-comparable test split noted   │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 09 │ 01_dataset_preparation.py   │ Uses existing SROIE split         │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 10 │ 02_train_donut.py           │ lm_head fix applied               │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 11 │ 03_train_trocr_yolo.py      │ Same 8 experiment combos          │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 12 │ 04_evaluate.py              │ Same test set, same metrics       │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│ 13 │ 05_compare_results.py       │ Comparison tables + plots work    │ □  │
├────┼─────────────────────────────┼──────────────────────────────────┼────┤
│    │ GLOBAL                      │ Train/test splits leak-free E2E   │ □  │
│    │ GLOBAL                      │ Constants deduplicated            │ □  │
│    │ GLOBAL                      │ GPU cleanup between experiments   │ □  │
│    │ GLOBAL                      │ No TODO/placeholder in any file   │ □  │
└────┴─────────────────────────────┴──────────────────────────────────┴────┘
```

---

## ═══ PHASE 13 — PR REQUIREMENTS ═══

**Format:** A single pull request from a new branch off current `main`.

**Non-negotiable PR requirements:**

1. Every fix must include an inline code comment explaining what was wrong and why the new implementation is correct.
2. No placeholders, no TODOs, no deferred fixes.
3. The PR description must include:
   - **"Verified Resolved"** — every known issue from Phase 1 with a one-sentence confirmation
   - **"Newly Discovered Issues"** — any additional problems found during the audit
   - The completed Pre-Ship Verification Checklist from Phase 12 with every box marked
4. `protobuf>=3.20.0` MUST be in requirements.txt (this alone caused 100% failure)
5. The pipeline must be runnable with: `git clone ... && pip install -r requirements.txt && python run_all.py`

---

## ═══ PHASE 14 — REPOSITORY CLEANUP ═══

- Remove stale experiment JSON files from repo root (they belong in `results/`)
- Remove committed log files (log1.txt, log2.txt, log3.txt) — these are runtime output
- Remove `experiment_1 (1).json` duplicate
- Remove dead code (`SROIENERLoader`)
- Add proper `.gitignore` entries for `results/`, `models/`, `*.log`
- Create `references.bib` (or remove dangling `\bibliography{references}` from paper.tex)

---

## ═══ SUMMARY: THE ONE COMMAND THAT MUST WORK ═══

```bash
git clone https://github.com/aiparallel0/kaggle.git && cd kaggle
pip install -r requirements.txt
python run_all.py
```

This must:
1. Download and split SROIE data (500/63/63)
2. Download and normalize 3 auxiliary datasets
3. Evaluate pretrained CORD baseline (zero-shot)
4. Run 8 DONUT experiments (train + evaluate each)
5. Run 8 TrOCR+YOLO experiments (train + evaluate each)
6. Generate comparison tables and plots
7. Fill paper.tex with all real metrics → paper_filled.tex
8. Print final summary with all results

Exit code 0. No crashes. No unresolved placeholders. No hardcoded metrics. Every number is real.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  END OF PROMPT                                                               ║
║  Previous PR merged — create new branch from main                            ║
║  The single blocking issue is protobuf>=3.20.0 in requirements.txt           ║
║  Everything else is accumulated technical debt + new TrOCR+YOLO pipeline     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```
