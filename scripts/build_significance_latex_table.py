#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from genai_detection.config import CONFIG

DEFAULT_METHOD_ORDER = [
    "on_the_fly_chatnoir",
    "on_the_fly_startpage",
    "in_domain",
    "two_step_llm",
    "ppmd",
    "unmasking",
    # "supervised_baseline",
    # "unsupervised_baseline_cosine",
    # "unsupervised_baseline_min-max",

]

# Comment out any metrics you do not want to include in the LaTeX output.
DEFAULT_METRICS = [
    # "accuracy",
    "recall",
    "precision",
    # "auroc_c_at_1",
    # "auroc",
    "c_at_1",
    # "f1",
]

STAR_MAP = {
    "0.05": "*",
    "0.01": "**",
    "0.005": "***",
}


def _dataset_display(name: str) -> str:
    return CONFIG.DATASET_TRANSLATIONS.get(name, name.replace("_", " ").title()) + " Dataset"


def _method_display(name: str) -> str:
    return CONFIG.LABEL_TRANSLATIONS.get(name, name.replace("_", " ").title())


def _metric_display(name: str) -> str:
    return CONFIG.SCORE_TRANSLATIONS.get(name, name)


def _read_metric_csv(path: Path) -> Dict[Tuple[str, str], str]:
    pairs: Dict[Tuple[str, str], str] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair = row.get("pair", "")
            min_level = row.get("min_significance_level", "").strip()
            if " vs " not in pair:
                continue
            left, right = pair.split(" vs ", 1)
            pairs[(left.strip(), right.strip())] = min_level
    return pairs


def _stars(min_level: str) -> str:
    if not min_level or min_level == "ns":
        return ""
    return STAR_MAP.get(min_level, "")


def _load_dataset_metrics(
    dataset_dir: Path,
    metrics: Iterable[str],
) -> Dict[str, Dict[Tuple[str, str], str]]:
    data: Dict[str, Dict[Tuple[str, str], str]] = {}
    for metric in metrics:
        csv_path = dataset_dir / "extracted_per_metric" / f"significance_{dataset_dir.name}_{metric}.csv"
        if not csv_path.exists():
            continue
        data[metric] = _read_metric_csv(csv_path)
    return data


def _pair_lookup(
    pairs: Dict[Tuple[str, str], str],
    a: str,
    b: str,
) -> str:
    if (a, b) in pairs:
        return pairs[(a, b)]
    if (b, a) in pairs:
        return pairs[(b, a)]
    return ""


def build_table(
    base_dir: Path,
    datasets: Iterable[str],
    metrics: Iterable[str],
    method_order: Iterable[str],
) -> str:
    method_list = list(method_order)
    metric_set = set(metrics)
    metric_list = [m for m in DEFAULT_METRICS if m in metric_set]
    if not metric_list:
        metric_list = list(DEFAULT_METRICS)
    ncols = 1 + len(method_list) * len(metric_list)

    header_groups = " & ".join(
        f"\\multicolumn{{{len(metric_list)}}}{{c}}{{\\textbf{{{_method_display(m)}}}}}"
        for m in method_list
    )
    header_metrics = " & ".join(_metric_display(m) for _ in method_list for m in metric_list)

    lines: List[str] = []
    lines.append("% Auto-generated; do not edit by hand.")
    lines.append("\\begin{table}[t]\\small")
    lines.append("  \\tabcolsep=0.11cm")
    lines.append("\\centering")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    col_groups = "|".join(["".join(["c"] * len(metric_list)) for _ in method_list])
    lines.append(f"\\begin{{tabular}}{{l|{col_groups}}}")
    lines.append("\\toprule")
    lines.append(f"\\textbf{{Method}} & {header_groups} \\\\")
    lines.append(f" & {header_metrics} \\\\")
    lines.append("\\midrule")

    for dataset in datasets:
        dataset_dir = base_dir / dataset
        if not dataset_dir.exists():
            continue
        metrics_data = _load_dataset_metrics(dataset_dir, metric_list)
        if not metrics_data:
            continue

        lines.append(f"\\multicolumn{{{ncols}}}{{@{{}}c@{{}}}}{{\\emph{{{_dataset_display(dataset)}}}}} \\\\")

        for i, method_row in enumerate(method_list):
            row_cells: List[str] = []
            for j, method_col in enumerate(method_list):
                if i == j:
                    row_cells.extend(["--"] * len(metric_list))
                    continue
                if j < i:
                    row_cells.extend([""] * len(metric_list))
                    continue
                for metric in metric_list:
                    pairs = metrics_data.get(metric, {})
                    level = _pair_lookup(pairs, method_row, method_col)
                    row_cells.append(_stars(level))
            lines.append(f"  {_method_display(method_row)} & " + " & ".join(row_cells) + " \\\\")

        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}}")
    lines.append(
        "\\caption{Corrected paired t-test based on 10-times 10-fold CV for the Blog Posts and Student Essays "
        "datasets. "
        "Asterisks indicate statistically significant differences "
        "({\\tiny \\texttt{*, **, ***}} for $p_{\\text{corr}} < 0.05, 0.01, 0.005$, respectively).}"
    )
    lines.append("\\label{tab:permutation-significance-stat-inf}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main() -> None:
    LOCAL_SAVE_PATH = (
        Path(__file__).resolve().parents[1] / CONFIG.SAVE_PATH / "reproduction"
    )
    print(f"Local save path: {LOCAL_SAVE_PATH}")
    parser = argparse.ArgumentParser(
        description="Build a LaTeX table from extracted per-metric significance CSVs."
    )
    parser.add_argument(
        "--base-dir",
        default=LOCAL_SAVE_PATH / "statistical_significance",
        help="Base directory containing dataset subdirectories.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets to include (default: all dataset directories under base dir).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics to include as columns (default: all in DEFAULT_METRICS).",
    )
    parser.add_argument(
        "--out-name",
        default="significance_table_all_metrics.tex",
        help="Output tex filename.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(p.name for p in base_dir.iterdir() if p.is_dir())

    latex = build_table(
        base_dir=base_dir,
        datasets=datasets,
        metrics=args.metrics,
        method_order=DEFAULT_METHOD_ORDER,
    )

    out_dir = base_dir / "to_tex"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    with open(out_path, "w") as f:
        f.write(latex)
        print(f"Wrote LaTeX table to {out_path}")


if __name__ == "__main__":
    main()
