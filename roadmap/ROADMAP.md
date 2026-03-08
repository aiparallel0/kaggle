# Research & Publication Roadmap

## Track 1 — Academic Paper Publication (Auto-Pilot)

### Goal
Bridge the gap between automated results generation (`paper_filled.tex`) and submission to recognised conferences/journals.

### Target Venues
- **Tier 1 (stretch):** CVPR, ICCV, ECCV (computer vision, receipt/document understanding)
- **Tier 2 (realistic):** ICDAR (International Conference on Document Analysis and Recognition) — directly on-topic
- **Tier 3 (rapid):** arXiv preprint → EMNLP/ACL workshop tracks on document NLP

### Pipeline Extension Plan
1. `paper_generator.py` → already generates `paper_filled.tex` with real metrics
2. **TODO:** Add `paper_publisher.py` module:
   - Auto-format to venue-specific LaTeX templates (CVPR, ICDAR)
   - Validate figure/table count against venue limits
   - Auto-generate camera-ready checklist
   - Export to arXiv submission package (`.tar.gz` with all assets)
3. **TODO:** Add `citation_manager.py`:
   - Auto-update related-work section when new DONUT/receipt-KIE papers appear (arXiv RSS)
   - Track our own citation count post-publication
4. **TODO:** `results/` → auto-generate ablation study tables comparing Exps 1–8

### Immediate Actions
- [ ] Write abstract + intro section stubs in `paper.tex`
- [ ] Ensure Exps 2–4 (naïve baselines) run successfully (fixed in this PR)
- [ ] Generate complete 8-experiment comparison table
- [ ] Add statistical significance tests (paired bootstrap) for F1 comparisons

---

## Track 2 — Domain Extension: Medical & Agricultural AI Image Detection

### Goal
Generalise the DONUT fine-tuning pipeline to other document/image detection domains where AI vision models are applied.

### Target Domains

#### Medical
- **Prescription OCR** — extract drug name, dosage, date, prescriber (analogous to SROIE fields)
  - Dataset candidates: CORD-medical, RxNorm, custom pharmacy receipt scans
  - New tokens: `<s_drug>`, `<s_dosage>`, `<s_prescriber>`, `<s_date>`
- **Lab report KIE** — extract test name, result, reference range, units
- **Insurance claim forms** — structured field extraction

#### Agricultural
- **Crop disease detection** — DONUT or YOLO for leaf image classification + field notes extraction
  - Dataset: PlantVillage, iNaturalist crop disease subset
- **Harvest log OCR** — extract crop type, yield, date, field ID from handwritten/printed logs
- **Pesticide/fertiliser label KIE** — structured extraction from chemical product labels

### Architecture Generalisation Plan
1. **TODO:** Create `domain_configs/` directory with per-domain `constants_<domain>.py`:
   - `constants_medical.py` — FIELDS, NEW_TOKENS for prescription schema
   - `constants_agro.py` — FIELDS, NEW_TOKENS for crop log schema
2. **TODO:** Make `dataset_loaders.py` domain-agnostic:
   - Accept `field_schema: list[str]` parameter instead of hardcoding SROIE fields
   - `BaseDatasetLoader` already provides the right abstraction — extend it
3. **TODO:** Make `run_experiments.py` domain-parameterised:
   - `--domain sroie|medical|agro` CLI flag
   - Loads domain-specific constants and experiment configs
4. **TODO:** `paper_generator.py` domain templates for each field

### Immediate Actions
- [ ] Create `domain_configs/` directory stub
- [ ] Add `--domain` flag skeleton to `run_all.py` argparse (no-op initially)
- [ ] Document the 4-step process for adding a new domain in `CONTRIBUTING.md`
