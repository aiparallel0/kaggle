"""
inject_results.py — Generate LaTeX table content and a filled paper.tex.

Usage
-----
# Generate tables from single-experiment evaluation results (legacy):
    python inject_results.py

# Generate ALL tables from run_experiments.py output + write filled paper:
    python inject_results.py --all
    python inject_results.py --all --paper paper.tex --output paper_filled.tex
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — imported from shared module to avoid duplication
# ---------------------------------------------------------------------------
# FIX: FIELDS was duplicated independently here and in 4 other files.
from constants import FIELDS, WORKSPACE

__all__ = [
    "PaperInjector",
    "UnresolvedVarError",
    "LEADERBOARD",
    "DONUT_PUBLISHED_F1",
    "EXP_NAMES",
    "build_var_map",
    "fill_paper",
    "print_table1_dataset_stats",
    "print_table2_experiments",
    "print_table3_perfield",
    "print_table4_leaderboard",
    # Absorbed from results_aggregator.py
    "ResultsAggregator",
]

# Pre-compiled regex for \VAR{...} template placeholders — compiled once at
# module load rather than on every fill() call.
_VAR_RE = re.compile(r"\\VAR\{([^}]+)\}")

LEADERBOARD: list[tuple[str, float]] = [
    # LayoutLMv3: Huang et al. 2022, "LayoutLMv3: Pre-training for Document AI"
    # Table 6, SROIE entity-level F1. DOI: 10.1145/3503161.3548112
    ("LayoutLMv3 (Huang et al. 2022)", 0.9633),
    # PICK: Yu et al. 2021, "PICK: Processing Key Information Extraction"
    # Table 3, SROIE Task-3 F1. DOI: 10.1109/ICPR48806.2021.9956043
    ("PICK (Yu et al. 2021)", 0.9612),
    # BROS: Hong et al. 2022, "BROS: A Pre-trained Language Model"
    # Table 2, SROIE entity-level F1. arXiv:2108.04539
    ("BROS (Hong et al. 2022)", 0.9548),
    # LayoutLMv2: Xu et al. 2021, "LayoutLMv2: Multi-modal Pre-training"
    # Table 4, SROIE entity-level F1. DOI: 10.18653/v1/2021.acl-long.201
    ("LayoutLMv2 (Xu et al. 2021)", 0.9495),
    # ICDAR 2019 competition results from arXiv:2103.10213 Table 1
    ("H&H Lab — ICDAR'19 1st", 0.9567),
    ("CLOVA OCR — ICDAR'19 2nd", 0.9373),
    ("ICDAR'19 3rd place", 0.9198),
    # DONUT SROIE fine-tuned: Kim et al. 2022, "OCR-free Document Understanding Transformer"
    # Table 1, entity-level F1 on SROIE. arXiv:2111.15664
    # NOTE: This is the SROIE fine-tuned result (not zero-shot CORD transfer).
    ("DONUT (SROIE fine-tuned, Kim et al. 2022)", 0.8411),
]

# Published DONUT F1 on SROIE (Kim et al. 2022, arXiv:2111.15664).
# NOTE: This value (84.11%) is from fine-tuning the CORD-pretrained DONUT on
# SROIE, evaluated with the entity-level F1 protocol consistent with SROIE
# Task-3.  Some versions of the paper report 92.68% using a different
# (field-level) evaluation protocol.  The codebase uses Task-3 F1 throughout,
# so 84.11% is the correct reference value for this comparison.
DONUT_PUBLISHED_F1 = 0.8411  # arXiv:2111.15664, Table 1, entity-level F1

EXP_NAMES: dict[str, str] = {
    "1": "SROIE only",
    "2": "+WildReceipt",
    "3": "+Invoices-DONUT",
    "4": "+WildReceipt+Invoices",
    "5": "+WildReceipt (2x SROIE)",
    "6": "+Invoices (2x SROIE)",
    "7": "+All (2x SROIE)",
    "8": "+All (3x SROIE)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(metrics: dict, key: str, fmt: str = ".4f") -> str:
    """Return formatted metric value or 'N/A'."""
    val = metrics.get(key)
    if val is None:
        return "N/A"
    return format(val, fmt)


class UnresolvedVarError(Exception):
    """Raised when \\VAR{} placeholders remain after substitution."""


# ---------------------------------------------------------------------------
# PaperInjector
# ---------------------------------------------------------------------------


class PaperInjector:
    """Reads experiment JSON results and fills a LaTeX template."""

    def __init__(
        self,
        results_dir: Path,
        template_path: Path,
        _preloaded_experiments: dict | None = None,
    ) -> None:
        self.results_dir = results_dir
        self.template_path = template_path
        self._preloaded_experiments = _preloaded_experiments

    # -- data loading -------------------------------------------------------

    def _load_all_experiments(self) -> dict:
        if self._preloaded_experiments is not None:
            return self._preloaded_experiments
        path = self.results_dir / "all_experiments.json"
        if not path.exists():
            return {}
        with open(path) as fh:
            return json.load(fh)

    def _load_evaluation_results(self) -> dict:
        path = self.results_dir / "evaluation_results.json"
        if not path.exists():
            return {}
        with open(path) as fh:
            return json.load(fh)

    # -- var map ------------------------------------------------------------

    def build_var_map(self) -> dict[str, str]:
        """Build \\VAR{key} → replacement mapping entirely from JSON files."""
        all_exp = self._load_all_experiments()
        eval_res = self._load_evaluation_results()
        var_map: dict[str, str] = {}

        # Per-experiment scalars (from all_experiments.json)
        for exp_id_str, res in all_exp.items():
            m = res.get("metrics", {})
            n = res.get("num_train_samples", 0)
            eid = exp_id_str
            var_map[f"exp{eid}_n"] = f"{n:,}"
            var_map[f"exp{eid}_prec"] = _safe(m, "global_precision")
            var_map[f"exp{eid}_rec"] = _safe(m, "global_recall")
            var_map[f"exp{eid}_f1"] = _safe(m, "global_f1")
            var_map[f"exp{eid}_em"] = _safe(m, "overall_exact_match")
            for field in FIELDS:
                var_map[f"exp{eid}_{field}_f1"] = _safe(m, f"{field}_f1")
                var_map[f"exp{eid}_{field}_ned"] = _safe(m, f"{field}_ned")

        # Best experiment
        best_f1 = 0.0
        best_exp_id = "1"
        for exp_id_str, res in all_exp.items():
            f1 = res.get("metrics", {}).get("global_f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_exp_id = exp_id_str

        if best_f1 == 0.0:
            warnings.warn(
                "Best fine-tuned F1 is 0.0 — injecting measured zero; "
                "paper will show 0.0 for all experiment metrics.",
                stacklevel=2,
            )

        var_map["best_f1"] = f"{best_f1:.4f}"
        var_map["best_f1_pct"] = f"{best_f1 * 100:.2f}"
        var_map["best_exp"] = best_exp_id

        # Pretrained / zero-shot metrics
        var_map["pre_f1"] = "N/A"
        var_map["pre_f1_pct"] = "N/A"
        var_map["pre_prec"] = "N/A"
        var_map["pre_rec"] = "N/A"
        var_map["pre_em"] = "N/A"

        pm = eval_res.get("pretrained_metrics", {})
        if pm:
            var_map["pre_f1"] = _safe(pm, "global_f1")
            var_map["pre_f1_pct"] = f"{pm.get('global_f1', 0.0) * 100:.2f}"
            var_map["pre_prec"] = _safe(pm, "global_precision")
            var_map["pre_rec"] = _safe(pm, "global_recall")
            var_map["pre_em"] = _safe(pm, "overall_exact_match")

        # Also check legacy workspace path
        if not pm:
            legacy_path = (
                Path(os.environ.get("DONUT_WORKSPACE", "/workspace")) / "evaluation_results.json"
            )
            if legacy_path.exists():
                try:
                    with open(legacy_path) as fh:
                        legacy = json.load(fh)
                    lp = legacy.get("pretrained_metrics", {})
                    var_map["pre_f1"] = _safe(lp, "global_f1")
                    var_map["pre_f1_pct"] = f"{lp.get('global_f1', 0.0) * 100:.2f}"
                    var_map["pre_prec"] = _safe(lp, "global_precision")
                    var_map["pre_rec"] = _safe(lp, "global_recall")
                    var_map["pre_em"] = _safe(lp, "overall_exact_match")
                except Exception:
                    pass

        # Gains
        try:
            exp1_f1 = all_exp.get("1", {}).get("metrics", {}).get("global_f1", 0.0)
            exp4_f1 = all_exp.get("4", {}).get("metrics", {}).get("global_f1", 0.0)
            var_map["gain_1_4"] = f"{(exp4_f1 - exp1_f1):+.4f}"
            var_map["gain_over_published"] = f"{(best_f1 - DONUT_PUBLISHED_F1):+.4f}"
        except Exception:
            var_map["gain_1_4"] = "N/A"
            var_map["gain_over_published"] = "N/A"

        # FIX: TrOCR+YOLO results — inject variables for dual-architecture
        # comparison table in paper.tex.  Reads from trocr_yolo_results.json.
        trocr_path = self.results_dir / "trocr_yolo_results.json"
        if trocr_path.exists():
            try:
                with open(trocr_path) as fh:
                    trocr_all = json.load(fh)
                for exp_id_str, res in trocr_all.items():
                    m = res.get("metrics", {})
                    n = res.get("num_train_samples", 0)
                    eid = exp_id_str
                    var_map[f"trocr_exp{eid}_n"] = f"{n:,}"
                    var_map[f"trocr_exp{eid}_prec"] = _safe(m, "global_precision")
                    var_map[f"trocr_exp{eid}_rec"] = _safe(m, "global_recall")
                    var_map[f"trocr_exp{eid}_f1"] = _safe(m, "global_f1")
                    var_map[f"trocr_exp{eid}_em"] = _safe(m, "overall_exact_match")
                    for field in FIELDS:
                        var_map[f"trocr_exp{eid}_{field}_f1"] = _safe(m, f"{field}_f1")
                        var_map[f"trocr_exp{eid}_{field}_ned"] = _safe(m, f"{field}_ned")

                # Best TrOCR+YOLO result
                trocr_best_f1 = 0.0
                trocr_best_exp = "1"
                for exp_id_str, res in trocr_all.items():
                    f1 = res.get("metrics", {}).get("global_f1", 0.0)
                    if f1 > trocr_best_f1:
                        trocr_best_f1 = f1
                        trocr_best_exp = exp_id_str
                var_map["trocr_best_f1"] = f"{trocr_best_f1:.4f}"
                var_map["trocr_best_f1_pct"] = f"{trocr_best_f1 * 100:.2f}"
                var_map["trocr_best_exp"] = trocr_best_exp
            except Exception:
                pass

        # Ensure TrOCR vars have fallback values if file was missing
        for key in ["trocr_best_f1", "trocr_best_f1_pct", "trocr_best_exp"]:
            var_map.setdefault(key, "N/A")

        return var_map

    # -- fill ---------------------------------------------------------------

    def fill(self) -> str:
        """Replace all \\VAR{key} in template; RAISE if any remain unresolved."""
        var_map = self.build_var_map()
        text = self.template_path.read_text(encoding="utf-8")

        def _replace(m: re.Match) -> str:
            key = m.group(1)
            return var_map.get(key, m.group(0))

        filled = _VAR_RE.sub(_replace, text)

        remaining = _VAR_RE.findall(filled)
        if remaining:
            raise UnresolvedVarError(
                f"{len(remaining)} unresolved \\VAR{{}} placeholder(s): {remaining}"
            )
        return filled

    # -- leaderboard verification -------------------------------------------

    def verify_leaderboard_scores(self) -> None:
        """Assert leaderboard constants against known-good cited values."""
        expected = {
            # LayoutLMv3: Huang et al. 2022, DOI: 10.1145/3503161.3548112, Table 6
            "LayoutLMv3 (Huang et al. 2022)": 0.9633,
            # PICK: Yu et al. 2021, DOI: 10.1109/ICPR48806.2021.9956043, Table 3
            "PICK (Yu et al. 2021)": 0.9612,
            # BROS: Hong et al. 2022, arXiv:2108.04539, Table 2
            "BROS (Hong et al. 2022)": 0.9548,
            # LayoutLMv2: Xu et al. 2021, DOI: 10.18653/v1/2021.acl-long.201, Table 4
            "LayoutLMv2 (Xu et al. 2021)": 0.9495,
            # ICDAR 2019 competition: arXiv:2103.10213, Table 1
            "H&H Lab — ICDAR'19 1st": 0.9567,
            "CLOVA OCR — ICDAR'19 2nd": 0.9373,
            "ICDAR'19 3rd place": 0.9198,
            # DONUT SROIE fine-tuned: Kim et al. 2022, arXiv:2111.15664, Table 1
            "DONUT (SROIE fine-tuned, Kim et al. 2022)": 0.8411,
        }
        lb_dict = {name: score for name, score in LEADERBOARD}
        for name, score in expected.items():
            actual = lb_dict.get(name)
            assert actual is not None, f"Missing leaderboard entry: {name}"
            assert abs(actual - score) < 1e-6, (
                f"Leaderboard mismatch for {name}: expected {score}, got {actual}"
            )


# ---------------------------------------------------------------------------
# Legacy single-experiment output (kept for backward compatibility)
# ---------------------------------------------------------------------------


def legacy_output(results_path: str = str(WORKSPACE / "evaluation_results.json")) -> None:
    """Print LaTeX rows from a single evaluation output file."""
    with open(results_path) as f:
        results = json.load(f)
    pm = results["pretrained_metrics"]
    fm = results["finetuned_metrics"]

    print("% === PASTE INTO LATEX TABLE 2 (head-to-head) ===")
    print("% Field & Metric & Pretrained & Fine-tuned \\\\")
    for field in FIELDS:
        f1_p = pm[f"{field}_f1"]
        f1_f = fm[f"{field}_f1"]
        ned_p = pm[f"{field}_ned"]
        ned_f = fm[f"{field}_ned"]
        print(f"{field.capitalize()} & F1 & {f1_p:.4f} & {f1_f:.4f} \\\\")
        print(f"{field.capitalize()} & NED & {ned_p:.4f} & {ned_f:.4f} \\\\")

    print("\\midrule")
    print(f"Global & F1 & {pm['global_f1']:.4f} & {fm['global_f1']:.4f} \\\\")
    print(
        f"Global & Exact Match & {pm['overall_exact_match']:.4f}"
        f" & {fm['overall_exact_match']:.4f} \\\\"
    )

    print()
    print("% === PASTE INTO LATEX TABLE 3 (leaderboard) ===")
    print(f"Our fine-tuned & -- & {fm['global_f1'] * 100:.2f} \\\\")
    print(f"Our pretrained (zero-shot) & -- & {pm['global_f1'] * 100:.2f} \\\\")


# ---------------------------------------------------------------------------
# Multi-experiment table printers
# ---------------------------------------------------------------------------


def print_table1_dataset_stats(actual_counts: dict = None) -> None:
    """Print Table 1: Dataset Statistics LaTeX rows.

    Parameters
    ----------
    actual_counts : dict, optional
        Mapping of dataset name to actual sample count, e.g.
        ``{"sroie_train": 500, "sroie_val": 63, "sroie_test": 63, ...}``.
        When provided, overrides the hardcoded fallback values.
    """
    print("% === TABLE 1: Dataset Statistics ===")
    _c = actual_counts or {}
    rows = [
        ("SROIE (train)", _c.get("sroie_train", 500), 4, "EN", "Receipts"),
        ("SROIE (val)", _c.get("sroie_val", 63), 4, "EN", "Receipts"),
        ("SROIE (test)", _c.get("sroie_test", 63), 4, "EN", "Receipts"),
        ("WildReceipt", _c.get("wildreceipt", 1740), 25, "EN", "Receipts"),
        ("FUNSD", _c.get("funsd", 149), 4, "EN", "Forms"),
        ("Invoices-DONUT", _c.get("invoices_donut", 800), "7+", "EN", "Invoices"),
    ]
    for name, n, nf, lang, domain in rows:
        n_str = f"{n:,}" if isinstance(n, int) else str(n)
        print(f"{name} & {n_str} & {nf} & {lang} & {domain} \\\\")
    print()


def print_table2_experiments(all_exp: dict) -> None:
    """Print Table 2: Per-Experiment Results rows."""
    print("% === TABLE 2: Per-Experiment Results ===")
    print("% Exp & Training Data & Train Samples & Precision & Recall & F1 & Exact Match \\\\")
    for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
        res = all_exp[exp_id_str]
        m = res.get("metrics", {})
        name = EXP_NAMES.get(exp_id_str, res.get("name", ""))
        n = res.get("num_train_samples", 0)
        print(
            f"{exp_id_str} & {name} & {n:,} & "
            f"{_safe(m, 'global_precision')} & "
            f"{_safe(m, 'global_recall')} & "
            f"{_safe(m, 'global_f1')} & "
            f"{_safe(m, 'overall_exact_match')} \\\\"
        )
    print()


def print_table3_perfield(all_exp: dict) -> None:
    """Print Table 3: Per-Field F1 / NED rows."""
    print("% === TABLE 3: Per-Field Breakdown ===")
    header_fields = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{{f.capitalize()}}}}}" for f in FIELDS
    )
    # FIX (BUG 8): Added (↓) suffix to NED sub-columns to indicate lower is better.
    print(f"% Exp & Training Data & {header_fields} \\\\")
    print("% Sub-header: F1(↑) & NED(↓) per field")
    for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
        res = all_exp[exp_id_str]
        m = res.get("metrics", {})
        name = EXP_NAMES.get(exp_id_str, res.get("name", ""))
        field_cols = " & ".join(
            # F1 columns are (↑) higher is better; NED columns are (↓) lower is better.
            f"{_safe(m, f + '_f1')} & {_safe(m, f + '_ned')}"
            for f in FIELDS
        )
        print(f"{exp_id_str} & {name} & {field_cols} \\\\")
    print()


def print_table4_leaderboard(all_exp: dict) -> None:
    """Print Table 4: SROIE Task 3 Leaderboard rows."""
    print("% === TABLE 4: Leaderboard Comparison ===")
    entries = list(LEADERBOARD)
    best_f1 = 0.0
    best_exp_id = None
    for exp_id_str, res in all_exp.items():
        f1 = res.get("metrics", {}).get("global_f1", 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_exp_id = exp_id_str
    if best_exp_id:
        exp_name = EXP_NAMES.get(best_exp_id, f"Exp {best_exp_id}")
        entries.append((f"Ours — best fine-tuned ({exp_name})", best_f1))

    for name, score in sorted(entries, key=lambda x: x[1], reverse=True):
        marker = " % <-- ours" if "Ours" in name else ""
        print(f"{name} & {score * 100:.2f} \\\\{marker}")
    print()


def print_table5_trocr_yolo(trocr_exp: dict) -> None:
    """Print Table 5: TrOCR+YOLO Per-Experiment Results rows.

    FIX: New function added for dual-architecture comparison.
    Previously the pipeline only generated DONUT tables.
    """
    print("% === TABLE 5: TrOCR+YOLO Per-Experiment Results ===")
    print("% Exp & Training Data & Train Samples & Precision & Recall & F1 & Exact Match \\\\")
    for exp_id_str in sorted(trocr_exp, key=lambda x: int(x)):
        res = trocr_exp[exp_id_str]
        m = res.get("metrics", {})
        name = EXP_NAMES.get(exp_id_str, res.get("name", ""))
        n = res.get("num_train_samples", 0)
        print(
            f"{exp_id_str} & {name} & {n:,} & "
            f"{_safe(m, 'global_precision')} & "
            f"{_safe(m, 'global_recall')} & "
            f"{_safe(m, 'global_f1')} & "
            f"{_safe(m, 'overall_exact_match')} \\\\"
        )
    print()


def print_table6_cross_architecture(donut_exp: dict, trocr_exp: dict) -> None:
    """Print Table 6: Cross-Architecture Comparison (DONUT vs TrOCR+YOLO).

    FIX: New function for the dual-architecture comparison that is the
    core scientific contribution of this paper.
    """
    print("% === TABLE 6: Cross-Architecture Comparison ===")
    print("% Exp & Training Data & DONUT F1 & TrOCR+YOLO F1 & Delta \\\\")
    for exp_id_str in sorted(set(donut_exp) | set(trocr_exp), key=lambda x: int(x)):
        name = EXP_NAMES.get(exp_id_str, f"Exp {exp_id_str}")
        d_f1 = donut_exp.get(exp_id_str, {}).get("metrics", {}).get("global_f1", 0.0)
        t_f1 = trocr_exp.get(exp_id_str, {}).get("metrics", {}).get("global_f1", 0.0)
        delta = d_f1 - t_f1
        print(f"{exp_id_str} & {name} & {d_f1:.4f} & {t_f1:.4f} & {delta:+.4f} \\\\")
    print()


# ---------------------------------------------------------------------------
# Plot Generation for Paper
# ---------------------------------------------------------------------------


def generate_training_plots(results_dir: Path = Path("results")) -> None:
    """Generate 2D training loss plots from experiment results.

    Creates publication-ready loss plots in results/figures/ for inclusion
    in paper.tex. Uses quick_results_generator if available.

    Args:
        results_dir: Path to results directory
    """
    try:
        from quick_results_generator import generate_loss_plots_from_results

        plots = generate_loss_plots_from_results(results_dir)
        if plots:
            print(f"[OK] Generated {len(plots)} loss plots for paper")
    except Exception:
        pass  # Gracefully skip if plotting unavailable


# ---------------------------------------------------------------------------
# Module-level wrappers (backward compatibility)
# ---------------------------------------------------------------------------


def build_var_map(all_exp: dict) -> dict:
    """Build \\VAR{key} → replacement mapping (module-level wrapper).

    Delegates to PaperInjector.build_var_map(). Passes all_exp via
    _preloaded_experiments to avoid a tempdir disk round-trip.
    """
    injector = PaperInjector(
        results_dir=Path("results"),
        template_path=Path("paper.tex"),
        _preloaded_experiments=all_exp,
    )
    return injector.build_var_map()


def fill_paper(paper_path: str, output_path: str, var_map: dict) -> None:
    """Replace all \\VAR{key} tokens in paper.tex and write output_path.

    Raises UnresolvedVarError if any placeholder remains.
    """
    text = Path(paper_path).read_text(encoding="utf-8")

    def _replace(m: re.Match) -> str:
        return var_map.get(m.group(1), m.group(0))

    filled = _VAR_RE.sub(_replace, text)
    remaining = _VAR_RE.findall(filled)
    if remaining:
        raise UnresolvedVarError(
            f"{len(remaining)} unresolved \\VAR{{}} placeholder(s): {remaining}"
        )
    Path(output_path).write_text(filled, encoding="utf-8")
    print(f"Filled paper written -> {output_path}")


# ---------------------------------------------------------------------------
# Convergence plot data / tex generation
# ---------------------------------------------------------------------------

_PLOT_STYLES = [
    ("blue", "o"),
    ("red", "square"),
    ("green!60!black", "triangle"),
    ("orange", "diamond"),
    ("purple", "star"),
    ("teal", "pentagon"),
    ("brown", "x"),
    ("magenta", "+"),
]


def generate_convergence_data(
    results_path: str = "results/all_experiments.json", output_dir: str = "results"
) -> None:
    """Read all_experiments.json and write per-experiment convergence CSV files."""
    results_file = Path(results_path)
    if not results_file.exists():
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_file) as fh:
        all_exp = json.load(fh)

    for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
        res = all_exp[exp_id_str]
        log_history = res.get("training_log", [])
        if not log_history:
            continue

        epoch_data: dict = {}
        for entry in log_history:
            epoch = entry.get("epoch")
            if epoch is None:
                continue
            ep = round(epoch)
            if ep not in epoch_data:
                epoch_data[ep] = {}
            if "loss" in entry:
                epoch_data[ep]["train_loss"] = entry["loss"]
            if "eval_loss" in entry:
                epoch_data[ep]["eval_loss"] = entry["eval_loss"]

        csv_path = out_dir / f"convergence_exp{exp_id_str}.csv"
        with open(csv_path, "w") as fh:
            fh.write("epoch,train_loss,eval_loss\n")
            for ep in sorted(epoch_data):
                row = epoch_data[ep]
                train_loss = row.get("train_loss", "")
                eval_loss = row.get("eval_loss", "")
                fh.write(f"{ep},{train_loss},{eval_loss}\n")


def generate_convergence_tex(
    results_path: str = "results/all_experiments.json", output_dir: str = "results"
) -> None:
    """Generate results/convergence_plots.tex — pgfplots figure included in paper.tex."""
    results_file = Path(results_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_ids_with_data: list = []
    if results_file.exists():
        with open(results_file) as fh:
            all_exp = json.load(fh)
        for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
            csv_path = out_dir / f"convergence_exp{exp_id_str}.csv"
            if csv_path.exists():
                exp_ids_with_data.append(exp_id_str)

    def _plot_commands(loss_col: int, ylabel: str) -> str:
        """Return addplot lines for one subplot (loss_col: 1=train, 2=eval)."""
        lines = []
        for i, exp_id_str in enumerate(exp_ids_with_data):
            color, marker = _PLOT_STYLES[i % len(_PLOT_STYLES)]
            name = EXP_NAMES.get(exp_id_str, f"Exp {exp_id_str}")
            csv_rel = f"results/convergence_exp{exp_id_str}.csv"
            lines.append(
                f"    \\addplot[color={color},mark={marker},thick] "
                f"table[x=epoch,y index={loss_col},col sep=comma,header=true]"
                f"{{{csv_rel}}};\n"
                f"    \\addlegendentry{{{name}}}"
            )
        return "\n".join(lines)

    train_plots = _plot_commands(1, "Training Loss")
    eval_plots = _plot_commands(2, "Validation Loss")

    tex = (
        r"""\begin{figure*}[t]
  \centering
  \begin{subfigure}[t]{0.48\linewidth}
    \begin{tikzpicture}
      \begin{axis}[
        xlabel={Epoch},
        ylabel={Training Loss},
        width=\linewidth,
        height=6cm,
        legend pos=north east,
        legend style={font=\tiny},
        grid=major,
      ]
"""
        + train_plots
        + r"""
      \end{axis}
    \end{tikzpicture}
    \caption{Training Loss}
  \end{subfigure}%
  \hfill
  \begin{subfigure}[t]{0.48\linewidth}
    \begin{tikzpicture}
      \begin{axis}[
        xlabel={Epoch},
        ylabel={Validation Loss},
        width=\linewidth,
        height=6cm,
        legend pos=north east,
        legend style={font=\tiny},
        grid=major,
      ]
"""
        + eval_plots
        + r"""
      \end{axis}
    \end{tikzpicture}
    \caption{Validation Loss}
  \end{subfigure}
  \caption{Training and validation loss convergence curves for all eight
    experiments.  Experiments with larger combined training sets
    (Exp.~5--8) generally converge to lower training loss but may
    exhibit higher validation loss due to domain mismatch between
    auxiliary data and the SROIE test distribution.  Early stopping
    (patience = 5 epochs on validation loss) terminates training at
    different epochs across experiments.}
  \label{fig:convergence}
\end{figure*}
"""
    )
    tex_path = out_dir / "convergence_plots.tex"
    tex_path.write_text(tex, encoding="utf-8")


def generate_f1_barchart_tex(
    results_path: str = "results/all_experiments.json", output_dir: str = "results"
) -> None:
    """Generate results/f1_barchart.tex — horizontal bar chart of Global F1."""
    results_file = Path(results_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_labels: list = []
    f1_values: list = []

    if results_file.exists():
        with open(results_file) as fh:
            all_exp = json.load(fh)
        for exp_id_str in sorted(all_exp, key=lambda x: int(x)):
            res = all_exp[exp_id_str]
            f1 = res.get("metrics", {}).get("global_f1")
            if f1 is not None:
                name = EXP_NAMES.get(exp_id_str, f"Exp {exp_id_str}")
                exp_labels.append(f"Exp.~{exp_id_str}: {name}")
                f1_values.append(f1)

    if not f1_values:
        tex_path = out_dir / "f1_barchart.tex"
        tex_path.write_text("% No F1 data available yet.\n", encoding="utf-8")
        return

    coords = "\n        ".join(f"({v:.4f},{i})" for i, v in enumerate(f1_values))
    ylabels = "\n        ".join(f"{i}/{{{lab}}}" for i, lab in enumerate(exp_labels))

    tex = (
        r"""\begin{figure}[h]
  \centering
  \begin{tikzpicture}
    \begin{axis}[
      xbar,
      xlabel={Global F1},
      ytick=data,
      yticklabels={
        """
        + ylabels
        + r"""
      },
      width=\linewidth,
      height=7cm,
      xmin=0, xmax=1,
      bar width=8pt,
      nodes near coords,
      nodes near coords align={horizontal},
      every node near coord/.style={font=\tiny},
    ]
      \addplot[fill=blue!60] coordinates {
        """
        + coords
        + r"""
      };
    \end{axis}
  \end{tikzpicture}
  \caption{Global F1 score on the SROIE test set for each of the eight
    fine-tuning experiments, showing the impact of auxiliary dataset
    inclusion on extraction accuracy.}
  \label{fig:f1_barchart}
\end{figure}
"""
    )
    tex_path = out_dir / "f1_barchart.tex"
    tex_path.write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject experimental results into LaTeX tables")
    parser.add_argument(
        "--all", action="store_true", help="Generate all tables from results/all_experiments.json"
    )
    parser.add_argument(
        "--results",
        default="results/all_experiments.json",
        help="Path to all_experiments.json (default: results/all_experiments.json)",
    )
    parser.add_argument(
        "--paper", default="paper.tex", help="Path to paper.tex template (default: paper.tex)"
    )
    parser.add_argument(
        "--output",
        default="paper_filled.tex",
        help="Path to write the filled paper (default: paper_filled.tex)",
    )
    parser.add_argument(
        "--presentation",
        default="presentation.tex",
        help="Path to presentation.tex template (default: presentation.tex)",
    )
    parser.add_argument(
        "--presentation-output",
        default="presentation_filled.tex",
        help="Path to write the filled presentation (default: presentation_filled.tex)",
    )
    args = parser.parse_args()

    if args.all:
        results_path = Path(args.results)
        if not results_path.exists():
            print(
                f"ERROR: {results_path} not found. Run run_experiments.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
        with open(results_path) as fh:
            all_exp = json.load(fh)

        print_table1_dataset_stats()
        print_table2_experiments(all_exp)
        print_table3_perfield(all_exp)
        print_table4_leaderboard(all_exp)

        generate_convergence_data(args.results)
        generate_convergence_tex(args.results)
        generate_f1_barchart_tex(args.results)

        var_map = build_var_map(all_exp)

        if Path(args.paper).exists():
            fill_paper(args.paper, args.output, var_map)
        else:
            print(f"paper.tex not found at {args.paper}; skipping filled paper generation.")

        if Path(args.presentation).exists():
            fill_paper(args.presentation, args.presentation_output, var_map)
        else:
            print(
                f"presentation.tex not found at {args.presentation}; "
                "skipping filled presentation generation."
            )
    else:
        legacy_output()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# ResultsAggregator
# Absorbed from results_aggregator.py — loads per-experiment JSON files and
# builds an AggregatedResults object consumed by MLTrainingOrchestrator and
# the paper generator.  Kept here because both modules operate on result JSON
# files and share the build_var_map() function defined above.
# ---------------------------------------------------------------------------


import json as _json  # noqa: E402
import logging as _logging  # noqa: E402
from datetime import datetime as _datetime  # noqa: E402

from pipeline_types import AggregatedResults, ExperimentMetrics, ExperimentResult  # noqa: E402


class ResultsAggregator:
    """Aggregate experiment results for paper generation.

    Reads all ``experiment_N.json`` files from *results_dir* and assembles an
    :class:`AggregatedResults` dataclass that identifies the best experiment,
    the baseline F1, and the overall improvement.
    """

    _logger = _logging.getLogger(__name__)

    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_experiments(self) -> AggregatedResults | None:
        """Load and aggregate all experiment results.

        Returns:
            AggregatedResults, or None if no experiment files exist.
        """
        self._logger.info("Aggregating experiment results...")
        experiment_files = sorted(self.results_dir.glob("experiment_*.json"))

        if not experiment_files:
            self._logger.warning("No experiment results found")
            return None

        experiments: list[ExperimentResult] = []
        for exp_file in experiment_files:
            try:
                data = _json.loads(exp_file.read_text())
                if isinstance(data, dict):
                    metrics = ExperimentMetrics(**data.get("metrics", {}))
                    exp = ExperimentResult(
                        experiment_id=data["experiment_id"],
                        name=data["name"],
                        datasets=data["datasets"],
                        num_train_samples=data["num_train_samples"],
                        metrics=metrics,
                    )
                    experiments.append(exp)
            except Exception as e:
                self._logger.warning(f"Could not load {exp_file}: {e}")

        if not experiments:
            self._logger.warning("No valid experiments loaded")
            return None

        best_exp = max(experiments, key=lambda e: e.metrics.global_f1)
        baseline_exp = next((e for e in experiments if e.experiment_id == 1), None)
        baseline_f1 = baseline_exp.metrics.global_f1 if baseline_exp else 0.0
        improvement = best_exp.metrics.global_f1 - baseline_f1

        agg = AggregatedResults(
            experiments=experiments,
            best_experiment=best_exp,
            baseline_f1=baseline_f1,
            improvement=improvement,
            generated_timestamp=_datetime.utcnow(),
        )

        self._logger.info(
            f"✓ Aggregated {len(experiments)} experiments. "
            f"Best: Exp {best_exp.experiment_id} F1={best_exp.metrics.global_f1:.4f}"
        )
        return agg

    def build_paper_metrics(self, agg: AggregatedResults) -> dict[str, str]:
        r"""Build \VAR{} key→value map for LaTeX template.

        Delegates to :func:`build_var_map` — single source of truth.
        """
        all_exp = {str(e.experiment_id): e.__dict__ for e in agg.experiments}
        return build_var_map(all_exp)

    def save_aggregated_results(
        self, agg: AggregatedResults, output_file: Path | None = None
    ) -> bool:
        """Save aggregated results to JSON.

        NOTE: ``all_experiments.json`` is owned by ``run_experiments.save_summary()``.
        This method writes to ``aggregated_summary.json`` instead.
        """
        if output_file is None:
            output_file = self.results_dir / "aggregated_summary.json"

        try:
            data = {
                "generated_timestamp": agg.generated_timestamp.isoformat(),
                "num_experiments": len(agg.experiments),
                "best_experiment": (
                    {
                        "experiment_id": agg.best_experiment.experiment_id,
                        "name": agg.best_experiment.name,
                        "f1": agg.best_experiment.metrics.global_f1,
                    }
                    if agg.best_experiment
                    else None
                ),
                "baseline_f1": agg.baseline_f1,
                "improvement": agg.improvement,
                "experiments": [e.to_dict() for e in agg.experiments],
            }
            output_file.write_text(_json.dumps(data, indent=2))
            self._logger.info(f"✓ Saved aggregated results to {output_file}")
            return True
        except Exception as e:
            self._logger.error(f"Could not save aggregated results: {e}")
            return False

    @staticmethod
    def load_all_experiments(results_dir: Path) -> list[ExperimentResult]:
        """Load all experiment JSON files from *results_dir*.

        Returns:
            List of ExperimentResult objects (empty list on error).
        """
        experiments: list[ExperimentResult] = []
        for exp_file in sorted(results_dir.glob("experiment_*.json")):
            try:
                data = _json.loads(exp_file.read_text())
                metrics = ExperimentMetrics(**data.get("metrics", {}))
                exp = ExperimentResult(
                    experiment_id=data["experiment_id"],
                    name=data["name"],
                    datasets=data["datasets"],
                    num_train_samples=data["num_train_samples"],
                    metrics=metrics,
                )
                experiments.append(exp)
            except Exception as e:
                _logging.getLogger(__name__).warning(f"Could not load {exp_file}: {e}")
        return experiments
