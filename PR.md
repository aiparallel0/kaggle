🔍 Deep Analysis: PR History → Parameter Issues
The Big Pattern (37 PRs → Recurring Problems)
Reading the PRs chronologically reveals 5 persistent failure categories that keep reoccurring and map directly to specific parameters in the control_suite:

🔴 Category 1: VRAM / OOM — The #1 Recurring Crisis
PRs involved: #78, #80, #82, #85, #88, #91, #92, #93, #96

These PRs fixed OOM crashes over and over. The root cause traces to 4 parameters set too aggressively for the GPU they're running on:

Parameter	Current Value	Problem	Recommended Options
yolo.batch	8	Fixed value OOMs on <16GB GPUs	4 (safe 12GB), 8 (16GB), 16 (24GB), 32 (40GB)
yolo.imgsz	512	PR #78 reduced from 640 to 512 for OOM — but the YOLO OOMs kept coming	320 (emergency), 416 (12GB), 512 (current), 640 (24GB+)
trocr.batch_size	16	PRs #80–#93 show repeated OOM with this size	4 (≤12GB), 8 (16GB), 16 (24GB), 32 (40GB+)
trocr.gradient_checkpointing	True	Correct — was missing before PR #82 and caused OOM	must stay True
trocr.use_cache	False	Correct — was breaking gradient_checkpointing before PR #101	must stay False
The real fix needed in control_suite — these should be dynamically computed, not hardcoded. Currently batch is hardcoded at 8 for YOLO and 16 for TrOCR regardless of VRAM. PRs #80, #91, #92, #93 patched this reactively each time.

🔴 Category 2: pytest Not Found — What It Means + Fix
Your error bash: pytest: command not found means pytest isn't in PATH. This is environment-specific. Fix options:

bash
# Option 1: Run via Python module (always works if pytest is installed)
python -m pytest tests/test_control_suite.py -v

# Option 2: Install pytest first
pip install pytest
pytest tests/test_control_suite.py -v

# Option 3: If in a venv that was just set up
source venv/bin/activate && python -m pytest tests/ -v

# Option 4: Check if it's installed at all
python -c "import pytest; print(pytest.__version__)"
The python -m pytest form is the most reliable across all environments — pytest binary is not always on PATH even when the package is installed.

🔴 Category 3: TrOCR date_f1=0 + Zero/Bad Results
PRs involved: #77, #83, #87, #88, #101

These trace to generation parameter mis-routing (PR #83 fixed model.config vs model.generation_config) and label tokenization mismatches (PR #101). The responsible control_suite parameters:

Parameter	Current Value	Problem	All Options
trocr.num_beams	4	Was set on wrong config object (model.config instead of generation_config)	1 (greedy, fastest), 2, 4 (current), 8 (best quality, slow)
trocr.no_repeat_ngram_size	0	Correct — was set to non-zero which blocked short OCR outputs	0 (disabled, ✅ correct for OCR), 2, 3 (harmful for short text)
trocr.max_length	128	PR #71 found benchmark used wrong max_length	64 (line crops), 128 (current), 256, 512 (full documents)
trocr.predict_with_generate	True	Must be True — was the root cause of date_f1=0 in PR #88	True (✅ mandatory for CER eval), False (teacher-forcing only)
🔴 Category 4: DONUT Convergence Failure
PR #88 was a big one: DONUT failed to converge at all. The parameters implicated:

Parameter	Current Value	PR Finding	All Options
donut.encoder_lr	5e-5	Layerwise LR was broken in PR #87 — decoder wasn't getting 2× the encoder LR	1e-5, 3e-5, 5e-5 (current), 1e-4
donut.decoder_lr	1e-4	Must be 2× encoder_lr	2e-5, 6e-5, 1e-4 (current = 2× encoder), 2e-4
donut.warmup_steps	40	Was set to 500 initially (PR #88 context) — exceeded total steps for small datasets → divergence	0, 20, 40 (current, safe for all exps), 100, 200, 500 (too large for Exp 1)
donut.sort_json_key	False	MUST be False — silently corrupts SROIE token ordering	False ✅ NEVER change
donut.tie_word_embeddings	False	MUST be False — causes F1=0 after checkpoint reload	False ✅ NEVER change
donut.sroie_oversample	1	Currently 1 (no oversampling) — CLAUDE.md says without 2× SROIE, auxiliary data hurts Exps 2–4	1 (current, baseline only), 2 (recommended for Exps 2–8)
⚠️ sroie_oversample=1 is currently WRONG for multi-dataset experiments. CLAUDE.md §8 explicitly states it must be ≥2 for experiments with auxiliary data (Exps 2–8). This is a real unfixed bug in the current defaults.

🔴 Category 5: YOLO-Specific Failures
PRs involved: #72, #73, #78, #87, #92, #96

Parameter	Current Value	PR Finding	All Options
yolo.freeze	None	Was completely missing from the training call (fixed in PR #106 — but value is still None)	None (all layers, current), 0 (same as None), 3 (freeze first 3 backbone layers, good start), 10 (freeze entire backbone), [0,1,2,3,4,5,6,7,8,9,10] (list form)
yolo.amp	True	PR #92 notes YOLO silently halves batch size on OOM when amp=True — explains inconsistent effective batch	True (current, enables auto-retry), False (disable if NaN losses)
yolo.close_mosaic	10	Was missing (PR #106 added it) — was relying on Ultralytics default	0 (disable, bad for accuracy), 5, 10 (current, Ultralytics default), 15, 20
yolo.rect	False	Must stay False — rect=True silently disables shuffle	False ✅ NEVER change (unless you know what you're doing)
yolo.cache	False	Potentially the cause of slow training (PR #100 context)	False (current, load from disk), "ram" (fastest, needs RAM), "disk" (cache preprocessed)
