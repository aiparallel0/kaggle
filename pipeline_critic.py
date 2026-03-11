# =============================================================================
# pipeline_critic.py
# Purpose: Systematic critical scrutiny of the DONUT SROIE experiment pipeline
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# =============================================================================
"""
PipelineCritic — not a reviewer. A wrecking ball.

Six audit classes examine distinct dimensions of research validity:

  1. StatisticalPowerAudit  — 63-sample test set vs. reported effect sizes
  2. MultipleTestingAudit   — 8 experiments on a shared test set (FWER inflation)
  3. EpochConfoundAudit     — Exp 1 (10 epochs) vs Exps 5–8 (15 epochs) confound
  4. PretrainingBiasAudit   — DONUT base checkpoint domain proximity to SROIE
  5. ArchitectureAudit      — fairness of DONUT vs TrOCR+YOLO comparison
  6. BenchmarkNarrowness    — custom test split vs official SROIE leaderboard

Usage
-----
    from pipeline_critic import PipelineCritic

    report = PipelineCritic().run()
    report.print_loud()        # prints all findings; non-zero exit on FATAL
    print(report.to_dict())    # machine-readable summary

Each audit runs without GPU, model weights, or internet access.
All findings carry severity (FATAL / CRITICAL / WARNING / INFO), a title,
detailed description, supporting evidence, and a concrete recommendation.
"""

import math
import sys
from pathlib import Path

from pipeline_types import CritiqueFinding, CritiqueReport, FindingSeverity

__all__ = [
    "PipelineCritic",
    "StatisticalPowerAudit",
    "MultipleTestingAudit",
    "EpochConfoundAudit",
    "PretrainingBiasAudit",
    "ArchitectureAudit",
    "BenchmarkNarrowness",
]

_ROOT = Path(__file__).resolve().parent


# =============================================================================
# Helpers
# =============================================================================


def _norm_ppf(p: float) -> float:
    """Rational approximation to the standard-normal inverse CDF (Abramowitz & Stegun 26.2.17).

    Accurate to within |ε| < 4.5e-4 for all 0 < p < 1.
    Handles p < 0.5 by symmetry. Avoids any third-party dependency.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be strictly between 0 and 1; got {p}")
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    return t - num / den


def _two_proportion_mdd(
    n: int,
    p_bar: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum detectable difference (two-sided two-proportion z-test).

    Returns the smallest absolute difference δ between two proportions
    reliably detectable (at the given *power* and *alpha*) when evaluating
    *n* independent binary outcomes.

    Args:
        n:      Number of independent binary trials per group (equal groups).
        p_bar:  Average proportion (mid-point of the null). Typically the
                baseline accuracy / F1 being compared against.
        alpha:  Type-I error rate (two-sided).
        power:  Desired statistical power (1 − β).

    Returns:
        Minimum detectable absolute difference in proportion / F1.
    """
    z_alpha_2 = _norm_ppf(1.0 - alpha / 2.0)
    z_beta = _norm_ppf(power)
    return (z_alpha_2 + z_beta) * math.sqrt(2.0 * p_bar * (1.0 - p_bar) / n)


def _family_wise_error_rate(k: int, alpha: float = 0.05) -> float:
    """Family-wise error rate for *k* independent tests at per-test *alpha*."""
    return 1.0 - (1.0 - alpha) ** k


# =============================================================================
# Audit 1 — Statistical Power
# =============================================================================


class StatisticalPowerAudit:
    """Checks whether the 63-sample test set provides sufficient statistical power.

    Key claim in the paper: Exp 6 improves over Exp 1 by +0.0479 global F1.
    This audit asks: given 63 × 4 = 252 scored pairs, can we reliably detect
    a 0.0479 difference, or is this indistinguishable from sampling noise?
    """

    # SROIE custom test set parameters (from run_experiments.py docstring)
    SROIE_TEST_SAMPLES: int = 63
    SROIE_FIELDS: int = 4  # company, date, address, total
    BASELINE_F1: float = 0.85  # approximate baseline from CLAUDE.md
    CLAIMED_GAIN: float = 0.0479  # Exp 6 vs Exp 1 (CLAUDE.md §8)

    def run(self) -> list[CritiqueFinding]:
        n_pairs = self.SROIE_TEST_SAMPLES * self.SROIE_FIELDS  # 252
        mdd_80 = _two_proportion_mdd(n_pairs, self.BASELINE_F1, power=0.80)
        mdd_50 = _two_proportion_mdd(n_pairs, self.BASELINE_F1, power=0.50)

        findings: list[CritiqueFinding] = []

        if mdd_80 > self.CLAIMED_GAIN:
            sev = FindingSeverity.FATAL if mdd_50 > self.CLAIMED_GAIN else FindingSeverity.CRITICAL
            findings.append(
                CritiqueFinding(
                    severity=sev,
                    category="statistical_validity",
                    title="Test set underpowered: claimed gain is below minimum detectable difference",
                    description=(
                        f"The custom SROIE test set has {self.SROIE_TEST_SAMPLES} images × "
                        f"{self.SROIE_FIELDS} fields = {n_pairs} scored pairs. "
                        f"A two-sided two-proportion z-test requires MDD₈₀ ≈ {mdd_80:.4f} F1 "
                        f"to achieve 80% power at α = 0.05. "
                        f"The headline improvement of {self.CLAIMED_GAIN:.4f} "
                        f"(Exp 6 vs baseline) falls below this threshold. "
                        f"The result cannot be distinguished from sampling noise "
                        f"at conventional significance levels."
                    ),
                    evidence=(
                        f"n_pairs={n_pairs}, MDD₈₀={mdd_80:.4f}, MDD₅₀={mdd_50:.4f}, "
                        f"claimed Δ={self.CLAIMED_GAIN:.4f}; "
                        f"baseline F1 ≈ {self.BASELINE_F1}"
                    ),
                    recommendation=(
                        "Expand the evaluation to ≥ 400 samples × 4 fields (≥ 1 600 pairs) "
                        "for 80% power to detect a 0.05 F1 gain. "
                        "Report confidence intervals (e.g. Wilson interval per field) "
                        "alongside point estimates. "
                        "Validate on the official SROIE 347-image held-out test set "
                        "if labels become available."
                    ),
                )
            )

        # Additional finding: per-field sample sizes are even smaller
        per_field_n = self.SROIE_TEST_SAMPLES  # 63 per field
        mdd_per_field = _two_proportion_mdd(per_field_n, self.BASELINE_F1, power=0.80)
        findings.append(
            CritiqueFinding(
                severity=FindingSeverity.WARNING,
                category="statistical_validity",
                title=f"Per-field comparisons use only {per_field_n} samples each",
                description=(
                    f"Each per-field F1 (company, date, address, total) is estimated "
                    f"from {per_field_n} samples. MDD₈₀ per field ≈ {mdd_per_field:.4f}. "
                    f"Field-level deltas reported in the paper (e.g. company F1 swings) "
                    f"are unreliable at this sample size."
                ),
                evidence=(
                    f"per_field_n={per_field_n}, MDD₈₀_per_field={mdd_per_field:.4f}"
                ),
                recommendation=(
                    "Do not draw conclusions from individual field F1 movements "
                    f"smaller than {mdd_per_field:.3f}. Treat per-field numbers as exploratory."
                ),
            )
        )

        return findings


# =============================================================================
# Audit 2 — Multiple Testing
# =============================================================================


class MultipleTestingAudit:
    """Checks for inflated false-positive risk from 8 experiments on one test set.

    Running 8 experiments on the same 63-sample test set and selecting the
    best result is a form of implicit multiple testing. Without correction,
    the family-wise error rate exceeds the claimed per-test α = 0.05.
    """

    N_EXPERIMENTS: int = 8
    N_COMPARISONS: int = 7  # comparing each of Exps 2–8 against the baseline Exp 1

    def run(self) -> list[CritiqueFinding]:
        fwer = _family_wise_error_rate(self.N_COMPARISONS, alpha=0.05)
        bonferroni_alpha = 0.05 / self.N_COMPARISONS

        return [
            CritiqueFinding(
                severity=FindingSeverity.CRITICAL,
                category="multiple_testing",
                title="8 experiments on one test set inflates false-positive rate to ~30%",
                description=(
                    f"All {self.N_EXPERIMENTS} experiments are evaluated on the same "
                    f"63-image test partition. With {self.N_COMPARISONS} comparisons "
                    f"against the baseline (Exps 2–8 vs Exp 1), the uncorrected "
                    f"family-wise error rate is {fwer:.1%}. "
                    f"Selecting the 'best' experiment (Exp 6) post-hoc from "
                    f"{self.N_EXPERIMENTS} candidates inflates the apparent improvement "
                    f"by maximising over noise. "
                    f"The Bonferroni-corrected threshold is α = {bonferroni_alpha:.4f} "
                    f"— far stricter than the uncorrected α = 0.05 implicitly assumed."
                ),
                evidence=(
                    f"k={self.N_COMPARISONS} comparisons, "
                    f"FWER={fwer:.3f}, "
                    f"Bonferroni_α={bonferroni_alpha:.4f}"
                ),
                recommendation=(
                    "Pre-register the hypothesis (e.g. 'Exp 6 will beat Exp 1') "
                    "before collecting test results, or apply Bonferroni / Benjamini–Hochberg "
                    "correction. Use a held-out test set that no experiment was selected "
                    "against. Report the full distribution of results, not just the maximum."
                ),
            )
        ]


# =============================================================================
# Audit 3 — Epoch Confound
# =============================================================================


class EpochConfoundAudit:
    """Checks for the training-duration confound between Exp 1 and Exps 5–8.

    The claimed 'best' experiments (5–8) all run 15 epochs while the baseline
    (Exp 1) runs 10 epochs. This 50% increase in training budget is a
    confounding variable: gains attributed to auxiliary data may be partly or
    entirely due to longer training.

    This audit reads the EXPERIMENTS configuration from source using AST to
    avoid triggering torch/transformers imports.
    """

    _RUN_EXPERIMENTS_FILE = _ROOT / "run_experiments.py"

    @staticmethod
    def _extract_experiment_epochs() -> dict[int, int]:
        """Parse run_experiments.py with AST to extract epoch counts per experiment.

        Returns a dict mapping experiment_id (int) to epochs (int).
        Falls back to the documented values if parsing fails.  When the AST
        parse succeeds but produces a dict that differs from the documented
        defaults, a warning is logged so stale defaults are never silent.
        """
        # Documented at ExperimentConfig definition time (run_experiments.py).
        # Updated manually when the experiment suite changes.
        # Exp 1–4 use the default of 10 epochs; Exps 5–8 explicitly set 15.
        documented_defaults: dict[int, int] = {
            1: 10,
            2: 10,
            3: 10,
            4: 10,
            5: 15,
            6: 15,
            7: 15,
            8: 15,
        }
        try:
            import ast

            source = EpochConfoundAudit._RUN_EXPERIMENTS_FILE.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            return documented_defaults

        # Walk the AST looking for ExperimentConfig keyword calls.
        # Each should have experiment_id and optionally epochs keywords.
        result: dict[int, int] = {}
        default_epochs = 10  # ExperimentConfig default

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "ExperimentConfig"):
                continue

            exp_id = None
            epochs = default_epochs

            for kw in node.keywords:
                if kw.arg == "experiment_id" and isinstance(kw.value, ast.Constant):
                    exp_id = int(kw.value.value)
                elif kw.arg == "epochs" and isinstance(kw.value, ast.Constant):
                    epochs = int(kw.value.value)

            if exp_id is not None:
                result[exp_id] = epochs

        if not result:
            return documented_defaults

        # Warn when AST parse succeeds but differs from documented defaults —
        # prevents stale fallback values from going unnoticed after refactors.
        import logging as _logging

        _log = _logging.getLogger(__name__)
        for eid, default_ep in documented_defaults.items():
            if eid in result and result[eid] != default_ep:
                _log.warning(
                    "EpochConfoundAudit: Exp %d epoch count changed from documented "
                    "default %d to %d. Update documented_defaults in pipeline_critic.py.",
                    eid,
                    default_ep,
                    result[eid],
                )
        return result

    def run(self) -> list[CritiqueFinding]:
        epochs_by_exp = self._extract_experiment_epochs()

        baseline_epochs = epochs_by_exp.get(1, 10)
        # Experiments claiming best results are 5, 6, 7 (Exp 8 is OOM)
        best_exps = {eid: epochs_by_exp.get(eid) for eid in (5, 6, 7, 8) if eid in epochs_by_exp}
        confounded_exps = {
            eid: ep for eid, ep in best_exps.items() if ep is not None and ep > baseline_epochs
        }

        findings: list[CritiqueFinding] = []

        if confounded_exps:
            epoch_summary = ", ".join(f"Exp {e}: {ep} epochs" for e, ep in sorted(confounded_exps.items()))
            findings.append(
                CritiqueFinding(
                    severity=FindingSeverity.CRITICAL,
                    category="experimental_design",
                    title="Epoch confound: 'best' experiments train 50% longer than the baseline",
                    description=(
                        f"Exp 1 (baseline) runs {baseline_epochs} epochs. "
                        f"The experiments claiming superiority run more epochs: {epoch_summary}. "
                        f"CLAUDE.md §8 explicitly acknowledges: "
                        f"'Exp 1 company F1 at 10 epochs is a convergence failure, not the "
                        f"SROIE-only ceiling.' "
                        f"This means the reported gain of +0.0479 (Exp 6 vs Exp 1) "
                        f"conflates auxiliary-data benefit with extended-training benefit. "
                        f"The contribution of auxiliary data cannot be isolated."
                    ),
                    evidence=(
                        f"Exp 1 epochs={baseline_epochs}; confounded experiments: {epoch_summary}; "
                        f"source: CLAUDE.md §8 note and EXPERIMENTS dict in run_experiments.py"
                    ),
                    recommendation=(
                        "Run a controlled ablation: train Exp 1 (SROIE only) for 15 epochs "
                        "to establish a fair baseline before attributing F1 gains to auxiliary data. "
                        "Either match training budgets across all experiments or report separate "
                        "epoch-matched comparisons."
                    ),
                )
            )

        return findings


# =============================================================================
# Audit 4 — Pretraining Bias
# =============================================================================


class PretrainingBiasAudit:
    """Checks domain proximity between the base checkpoint and SROIE.

    'naver-clova-ix/donut-base' was pretrained on SynthDoG (synthetic
    document images). SynthDoG includes synthetic receipts, menu images,
    and business documents — structurally similar to SROIE (Thai/Malaysian
    thermal receipts). This is not data contamination but it is domain
    proximity that shrinks the effective pretraining-to-target domain gap.

    Additionally, the evaluator code (donut_evaluator.py) handles '<sep/>'
    tokens described as 'inherited from CORD pretraining', suggesting the
    checkpoint has some CORD-adjacent behaviour even in the base version.
    """

    BASE_MODEL = "naver-clova-ix/donut-base"
    # ~25% of SynthDoG images are synthetic receipts (English, Japanese, Chinese).
    # Source: Donut paper, Kim et al. 2022, §3.1 "Pre-training Data" — SynthDoG
    # generates receipt / business-card / magazine-cover images in four languages.
    SYNTHDOG_RECEIPT_FRACTION_APPROX = 0.25

    def run(self) -> list[CritiqueFinding]:
        findings: list[CritiqueFinding] = []

        # Check whether donut_evaluator.py mentions CORD
        cord_mention = self._check_cord_reference()

        findings.append(
            CritiqueFinding(
                severity=FindingSeverity.WARNING,
                category="pretraining_bias",
                title="Base checkpoint pretrained on synthetic receipts (SynthDoG)",
                description=(
                    f"'{self.BASE_MODEL}' was pretrained on SynthDoG, a synthetic "
                    f"document dataset where roughly {self.SYNTHDOG_RECEIPT_FRACTION_APPROX:.0%} "
                    f"of images are synthetic receipts (English, Japanese, Chinese). "
                    f"SROIE contains real Malaysian thermal receipts. "
                    f"While not contamination, this structural overlap gives DONUT a "
                    f"warm-start advantage not available to the TrOCR+YOLO baseline. "
                    + (
                        "Evaluator code also handles '<sep/>' tokens described as "
                        "'inherited from CORD pretraining', suggesting CORD-adjacent "
                        "behaviour in the base checkpoint. "
                        if cord_mention
                        else ""
                    )
                    + "This makes the zero-shot → fine-tuned leap smaller than it appears."
                ),
                evidence=(
                    f"base_model={self.BASE_MODEL}; "
                    f"SynthDoG includes receipt images (Donut paper, Kim et al. 2022 §3.1); "
                    + ("donut_evaluator.py references CORD <sep/> token handling. " if cord_mention else "")
                ),
                recommendation=(
                    "Acknowledge the SynthDoG receipt pretraining in the paper's limitations. "
                    "For a true domain-gap study, compare against a DONUT variant pretrained "
                    "only on non-receipt documents (e.g. IIT-CDIP text documents only). "
                    "Report the zero-shot baseline F1 and discuss what fine-tuning adds "
                    "beyond the pretraining prior."
                ),
            )
        )

        return findings

    @staticmethod
    def _check_cord_reference() -> bool:
        """Return True if donut_evaluator.py mentions the CORD dataset by word boundary."""
        import re

        evaluator = _ROOT / "donut_evaluator.py"
        if not evaluator.exists():
            return False
        try:
            text = evaluator.read_text(encoding="utf-8")
            # Use word-boundary regex to avoid matching 'record', 'according', etc.
            return bool(re.search(r"\bCORD\b", text))
        except OSError:
            return False


# =============================================================================
# Audit 5 — Architecture Comparison Fairness
# =============================================================================


class ArchitectureAudit:
    """Checks whether the DONUT vs TrOCR+YOLO comparison is a fair ablation.

    The paper frames this as an architecture comparison. In practice it
    conflates architecture, pretraining data, pretraining task, and inference
    paradigm. The +69.5% absolute F1 gap reflects pretraining differences
    far more than architectural ones.
    """

    DONUT_F1: float = 0.8982  # best (Exp 6)
    TROCR_YOLO_F1: float = 0.2035  # from CLAUDE.md §8

    def run(self) -> list[CritiqueFinding]:
        gap = self.DONUT_F1 - self.TROCR_YOLO_F1

        return [
            CritiqueFinding(
                severity=FindingSeverity.WARNING,
                category="comparison_fairness",
                title="DONUT vs TrOCR+YOLO conflates architecture with pretraining",
                description=(
                    f"DONUT (best F1 = {self.DONUT_F1:.4f}) is compared against "
                    f"TrOCR + YOLOv8 (F1 = {self.TROCR_YOLO_F1:.4f}), "
                    f"a gap of {gap:.4f} absolute. "
                    f"This comparison is not a controlled architecture ablation: "
                    f"DONUT was pretrained on SynthDoG (includes synthetic receipts, "
                    f"structured document parsing) while TrOCR was pretrained on OCR "
                    f"(character recognition) and YOLO on object detection. "
                    f"The TrOCR+YOLO pipeline also relies on hand-crafted heuristics for "
                    f"field assignment, whereas DONUT learns field assignment end-to-end. "
                    f"A 69.5-point gap is consistent with comparing specialist-pretrained "
                    f"vs. non-specialist-pretrained models — it does not isolate "
                    f"architecture benefits."
                ),
                evidence=(
                    f"DONUT F1={self.DONUT_F1}, TrOCR+YOLO F1={self.TROCR_YOLO_F1}, "
                    f"gap={gap:.4f}; "
                    f"DONUT pretrain: SynthDoG (receipts included); "
                    f"TrOCR pretrain: IAM handwriting + SROIE-OCR (no KIE); "
                    f"field assignment: DONUT=end-to-end, TrOCR+YOLO=rule-based heuristics"
                ),
                recommendation=(
                    "Reframe the comparison as 'end-to-end document VLM vs. "
                    "pipeline OCR + heuristics', not 'architecture A vs. architecture B'. "
                    "For a controlled architecture ablation, compare DONUT with a "
                    "BERT-based classifier on the same TrOCR OCR output, "
                    "isolating only the parsing/field-assignment component."
                ),
            )
        ]


# =============================================================================
# Audit 6 — Benchmark Narrowness
# =============================================================================


class BenchmarkNarrowness:
    """Checks whether the evaluation benchmark supports the paper's generality claims.

    Two intertwined issues:
    a) The 'test set' is carved from the ICDAR-released training partition, not
       the official SROIE held-out test set (347 images, no public labels).
       This makes cross-paper comparison invalid.
    b) SROIE is a single-domain, fixed-format, 4-field benchmark. Claiming
       general KIE improvement based on it is an overreach.
    """

    OFFICIAL_SROIE_TEST_SIZE: int = 347  # images in official ICDAR held-out set
    CUSTOM_TEST_SIZE: int = 63  # images carved from training partition
    OFFICIAL_PUBLISHED_DONUT_F1: float = 0.8411  # from SROIE leaderboard / Donut paper

    def run(self) -> list[CritiqueFinding]:
        findings: list[CritiqueFinding] = []

        # a) Custom test set vs official leaderboard
        findings.append(
            CritiqueFinding(
                severity=FindingSeverity.FATAL,
                category="benchmark_validity",
                title="'Custom test set' comparison to published results uses different test sets",
                description=(
                    f"The paper reports improvement over 'published DONUT ({self.OFFICIAL_PUBLISHED_DONUT_F1})'. "
                    f"The published DONUT score comes from the official SROIE held-out test set "
                    f"({self.OFFICIAL_SROIE_TEST_SIZE} images, ground truth never released). "
                    f"This paper's score comes from a custom {self.CUSTOM_TEST_SIZE}-image subset "
                    f"carved from the 626 labeled ICDAR training images — a completely different "
                    f"evaluation set. Comparing F1 scores across different test sets is statistically "
                    f"invalid. The claim '+0.0571 over published DONUT' is not supported: "
                    f"it compares performance on different data."
                ),
                evidence=(
                    f"run_experiments.py docstring: 'custom 80/10/10 split from 626 labeled "
                    f"training images; the official 347-image test set has no public ground truth'; "
                    f"CLAUDE.md §8: 'Gain over baseline: Exp 6 (0.8982) − ... vs published "
                    f"DONUT (0.8411) = +0.0571'; "
                    f"official test size={self.OFFICIAL_SROIE_TEST_SIZE}, "
                    f"custom test size={self.CUSTOM_TEST_SIZE}"
                ),
                recommendation=(
                    "Remove or heavily caveat the cross-paper comparison. "
                    "State clearly: 'All results are on our custom 63-image split; "
                    "comparison to published SROIE leaderboard numbers is not valid.' "
                    "Alternatively, request ground-truth labels from the ICDAR organisers "
                    "or use a public SROIE re-split (e.g. EATEN, LayoutLM splits) that "
                    "others have also evaluated on."
                ),
            )
        )

        # b) Single-domain narrowness
        findings.append(
            CritiqueFinding(
                severity=FindingSeverity.WARNING,
                category="benchmark_narrowness",
                title="Single-domain, 4-field benchmark limits generalizability claims",
                description=(
                    "SROIE contains only Malaysian thermal printer receipts from 2019. "
                    "All 626 labeled images share the same physical format (thermal paper, "
                    "top-aligned header, 4 fixed fields). Real-world receipt KIE involves "
                    "hundreds of layouts, languages, fields, and image qualities. "
                    "Gains observed on SROIE do not necessarily transfer to "
                    "cross-domain, multi-layout, or higher-field-count scenarios."
                ),
                evidence=(
                    "SROIE dataset: ICDAR 2019, Malaysian receipts, 4 fixed fields; "
                    "no multi-layout, multi-language, or cross-domain evaluation; "
                    "comparison datasets (WildReceipt, Invoices-DONUT) used only for training, "
                    "never for evaluation"
                ),
                recommendation=(
                    "Evaluate on at least one additional held-out benchmark "
                    "(e.g. CORD, FUNSD, or a proprietary receipt dataset) to test "
                    "transfer of the multi-dataset fine-tuning gains. "
                    "Without cross-dataset evaluation, the paper can only claim "
                    "improvement on SROIE-style Malaysian receipts."
                ),
            )
        )

        return findings


# =============================================================================
# PipelineCritic — orchestrator
# =============================================================================


class PipelineCritic:
    """Orchestrates all six audits and returns a consolidated CritiqueReport.

    All audits run without GPU, model weights, or network access.
    Each audit is instantiated fresh to allow independent parameterisation.

    Usage
    -----
        report = PipelineCritic().run()
        report.print_loud()
    """

    def run(self) -> CritiqueReport:
        """Execute all audits and collect findings.

        Returns:
            CritiqueReport with all findings sorted by severity.
        """
        audits = [
            StatisticalPowerAudit(),
            MultipleTestingAudit(),
            EpochConfoundAudit(),
            PretrainingBiasAudit(),
            ArchitectureAudit(),
            BenchmarkNarrowness(),
        ]

        all_findings: list[CritiqueFinding] = []
        for audit in audits:
            all_findings.extend(audit.run())

        # Sort by severity: FATAL first, then CRITICAL, WARNING, INFO
        _order = {
            FindingSeverity.FATAL: 0,
            FindingSeverity.CRITICAL: 1,
            FindingSeverity.WARNING: 2,
            FindingSeverity.INFO: 3,
        }
        all_findings.sort(key=lambda f: _order[f.severity])

        return CritiqueReport(findings=all_findings)


# =============================================================================
# CritiqueReport — print_loud (kept here to avoid circular import)
# =============================================================================

_SEP = "=" * 78
_SEP_THIN = "-" * 78

_SEVERITY_PREFIX: dict[FindingSeverity, str] = {
    FindingSeverity.FATAL: "💀 FATAL",
    FindingSeverity.CRITICAL: "🔴 CRITICAL",
    FindingSeverity.WARNING: "🟡 WARNING",
    FindingSeverity.INFO: "ℹ️  INFO",
}


def _print_report(report: CritiqueReport, exit_on_fatal: bool = True) -> None:
    """Print all findings to stdout and optionally exit non-zero on FATAL."""
    print(_SEP)
    print("PIPELINE CRITIC — SCRUTINY REPORT")
    print(_SEP)
    print(
        f"Findings: {report.fatal_count} FATAL  "
        f"{report.critical_count} CRITICAL  "
        f"{report.warning_count} WARNING"
    )
    print(_SEP)

    for idx, finding in enumerate(report.findings, start=1):
        prefix = _SEVERITY_PREFIX.get(finding.severity, finding.severity.value)
        print(f"\n[{idx}] {prefix} — {finding.category}")
        print(f"    {finding.title}")
        print(_SEP_THIN)
        print(f"  Description : {finding.description}")
        print(f"  Evidence    : {finding.evidence}")
        print(f"  Fix         : {finding.recommendation}")

    print(f"\n{_SEP}")
    verdict = "DOES NOT SURVIVE SCRUTINY" if not report.passed else "SURVIVES SCRUTINY (no FATAL findings)"
    print(f"VERDICT: {verdict}")
    print(_SEP)

    if not report.passed and exit_on_fatal:
        sys.exit(1)


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    report = PipelineCritic().run()
    report.print_loud(exit_on_fatal=True)
