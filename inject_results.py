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

import argparse
import json
import re
import sys
from pathlib import Path

FIELDS = ["company", "date", "address", "total"]

LEADERBOARD = [
    # Post-competition SOTA (verified from published papers)
    ("LayoutLMv3 (Huang et al. 2022)", 0.9633),
    ("PICK (Yu et al. 2021)", 0.9612),
    ("BROS (Hong et al. 2022)", 0.9548),
    ("LayoutLMv2 (Xu et al. 2021)", 0.9495),
    # ICDAR 2019 original competition top-3 (from arxiv:2103.10213)
    ("H&H Lab — ICDAR'19 1st", 0.9567),
    ("CLOVA OCR — ICDAR'19 2nd", 0.9373),
    ("ICDAR'19 3rd place", 0.9198),
    # Published baselines
    # Zero-shot F1 from Table 1 of "OCR-free Document Understanding Transformer" (Kim et al., ECCV 2022)
    ("DONUT zero-shot (Kim et al. 2022)", 0.8411),
]

# Published DONUT zero-shot F1 on SROIE (Kim et al. 2022)
DONUT_ZEROSHOT_F1 = 0.8411

EXP_NAMES = {
    "1": "SROIE only",
    "2": "+WildReceipt",
    "3": "+SROIE-NER",
    "4": "+CORD",
    "5": "+WildReceipt+CORD",
    "6": "+SROIE-NER+CORD",
    "7": "+All",
    "8": "+Invoices-DONUT",
}


# ---------------------------------------------------------------------------
# Legacy single-experiment output (kept for backward compatibility)
# ---------------------------------------------------------------------------

def legacy_output(results_path: str = "/workspace/evaluation_results.json") -> None:
    """Print LaTeX rows from a single evaluate.py output file."""
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

    print(f"\\midrule")
    print(f"Global & F1 & {pm['global_f1']:.4f} & {fm['global_f1']:.4f} \\\\")
    print(f"Global & Exact Match & {pm['overall_exact_match']:.4f} & {fm['overall_exact_match']:.4f} \\\\")

    print()
    print("% === PASTE INTO LATEX TABLE 3 (leaderboard) ===")
    print(f"Our fine-tuned & -- & {fm['global_f1']*100:.2f} \\\\")
    print(f"Our pretrained (zero-shot) & -- & {pm['global_f1']*100:.2f} \\\\")


# ---------------------------------------------------------------------------
# Multi-experiment output
# ---------------------------------------------------------------------------

def _safe(metrics: dict, key: str, fmt: str = ".4f") -> str:
    """Return formatted metric value or 'N/A'."""
    val = metrics.get(key)
    if val is None:
        return "N/A"
    return format(val, fmt)


def print_table1_dataset_stats(actual_counts: dict = None) -> None:
    """Print Table 1: Dataset Statistics LaTeX rows.

    Parameters
    ----------
    actual_counts : dict, optional
        Mapping of dataset name to actual sample count, e.g.
        ``{"sroie_train": 526, "sroie_test": 100, ...}``.
        When provided, overrides the hardcoded fallback values.
    """
    print("% === TABLE 1: Dataset Statistics ===")
    # Fallback values match the expected split (526 train + 100 test) defined
    # in stage_install(); actual counts passed from run_all.py when available.
    _c = actual_counts or {}
    rows = [
        ("SROIE (train)",    _c.get("sroie_train",    526),    4,    "EN",    "Receipts"),
        ("SROIE (test)",     _c.get("sroie_test",     100),    4,    "EN",    "Receipts"),
        ("WildReceipt",      _c.get("wildreceipt",   1740),   25,   "EN",    "Receipts"),
        ("SROIE-NER",        _c.get("sroie_ner",      526),    4,   "EN",    "Receipts"),
        ("CORD v2",          _c.get("cord",            900),  "30+", "ID",    "Receipts"),
        ("Invoices-DONUT",   _c.get("invoices_donut",  800),  "7+",  "EN",    "Invoices"),
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
            # FIX (BUG 8): NED direction indicator preserved in column ordering comments.
            # F1 columns are (↑) higher is better; NED columns are (↓) lower is better.
            f"{_safe(m, f + '_f1')} & {_safe(m, f + '_ned')}" for f in FIELDS
        )
        print(f"{exp_id_str} & {name} & {field_cols} \\\\")
    print()


def print_table4_leaderboard(all_exp: dict) -> None:
    """Print Table 4: SROIE Task 3 Leaderboard rows."""
    print("% === TABLE 4: Leaderboard Comparison ===")
    entries = list(LEADERBOARD)
    # Find best fine-tuned result
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
        print(f"{name} & {score*100:.2f} \\\\{marker}")
    print()


def build_var_map(all_exp: dict) -> dict:
    """Build a mapping from \\VAR{key} → replacement string."""
    var_map = {}

    # Per-experiment scalars
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
    var_map["best_f1"] = f"{best_f1:.4f}"
    var_map["best_f1_pct"] = f"{best_f1*100:.2f}"
    var_map["best_exp"] = best_exp_id

    # Pretrained (zero-shot CORD) placeholder — filled if available
    var_map["pre_f1"] = "N/A"
    var_map["pre_f1_pct"] = "N/A"
    var_map["pre_prec"] = "N/A"
    var_map["pre_rec"] = "N/A"
    var_map["pre_em"] = "N/A"

    # Try to read pretrained metrics from legacy evaluate output
    legacy_path = Path("/workspace/evaluation_results.json")
    if legacy_path.exists():
        try:
            with open(legacy_path) as fh:
                legacy = json.load(fh)
            pm = legacy.get("pretrained_metrics", {})
            var_map["pre_f1"] = _safe(pm, "global_f1")
            var_map["pre_f1_pct"] = f"{pm.get('global_f1', 0.0)*100:.2f}"
            var_map["pre_prec"] = _safe(pm, "global_precision")
            var_map["pre_rec"] = _safe(pm, "global_recall")
            var_map["pre_em"] = _safe(pm, "overall_exact_match")
        except Exception:
            pass

    # Gains
    try:
        exp1_f1 = all_exp.get("1", {}).get("metrics", {}).get("global_f1", 0.0)
        exp4_f1 = all_exp.get("4", {}).get("metrics", {}).get("global_f1", 0.0)
        var_map["gain_1_4"] = f"{(exp4_f1 - exp1_f1):+.4f}"
        var_map["gain_over_published"] = f"{(best_f1 - DONUT_ZEROSHOT_F1):+.4f}"
    except Exception:
        var_map["gain_1_4"] = "N/A"
        var_map["gain_over_published"] = "N/A"

    return var_map


def fill_paper(paper_path: str, output_path: str, var_map: dict) -> None:
    """Replace all \\VAR{key} tokens in paper.tex and write output_path."""
    text = Path(paper_path).read_text(encoding="utf-8")

    def replace(m):
        key = m.group(1)
        return var_map.get(key, m.group(0))  # leave unknown vars unchanged

    filled = re.sub(r"\\VAR\{([^}]+)\}", replace, text)
    Path(output_path).write_text(filled, encoding="utf-8")
    print(f"Filled paper written → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inject experimental results into LaTeX tables")
    parser.add_argument("--all", action="store_true",
                        help="Generate all tables from results/all_experiments.json")
    parser.add_argument("--results", default="results/all_experiments.json",
                        help="Path to all_experiments.json (default: results/all_experiments.json)")
    parser.add_argument("--paper", default="paper.tex",
                        help="Path to paper.tex template (default: paper.tex)")
    parser.add_argument("--output", default="paper_filled.tex",
                        help="Path to write the filled paper (default: paper_filled.tex)")
    args = parser.parse_args()

    if args.all:
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"ERROR: {results_path} not found. Run run_experiments.py first.", file=sys.stderr)
            sys.exit(1)
        with open(results_path) as fh:
            all_exp = json.load(fh)

        print_table1_dataset_stats()
        print_table2_experiments(all_exp)
        print_table3_perfield(all_exp)
        print_table4_leaderboard(all_exp)

        if Path(args.paper).exists():
            var_map = build_var_map(all_exp)
            fill_paper(args.paper, args.output, var_map)
        else:
            print(f"paper.tex not found at {args.paper}; skipping filled paper generation.")
    else:
        # Legacy mode: read /workspace/evaluation_results.json
        legacy_output()


if __name__ == "__main__":
    main()