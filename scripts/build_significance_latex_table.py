#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from genai_detection.config import CONFIG

DEFAULT_METHOD_ORDER = [
    # "on_the_fly_chatnoir",
    # "on_the_fly_startpage",
    # "two_step_llm",
    "in_domain",

    # "ppmd",
    # "unmasking",
    # "supervised_baseline",
    # "unsupervised_baseline_cosine",
    # "unsupervised_baseline_min-max",

    "bdi",
    "homotopy",
    "potha2017",
    "asgalf",
    "std_impostor",
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
    # "0.1": "*",
    "0.05": "*",
    "0.01": "**",
    "0.005": "***",
}


def _dataset_display(name: str) -> str:
    return CONFIG.DATASET_TRANSLATIONS.get(name, name.replace("_", " ").title()) + " Dataset"


def _method_display(name: str) -> str:
    return CONFIG.LABEL_TRANSLATIONS.get(name, name.replace("_", " ").title()).replace("&", r"\&")


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
    dataset_name: str
) -> Dict[str, Dict[Tuple[str, str], str]]:
    data: Dict[str, Dict[Tuple[str, str], str]] = {}
    for metric in metrics:
        # reproduction/pan_metrics/statistical_significance/extracted_per_metric
        csv_path = dataset_dir / f"significance_{dataset_name}_{metric}.csv"
        if not csv_path.exists():
            print(f"[{dataset_name}]: No significance CSV found for {metric} in {csv_path}. Run `python3 "
                  f"scripts/extract_statistical_significance_tables.py` first. Skipping this metric.")
            continue
        data[metric] = _read_metric_csv(csv_path)
    return data


def _pair_lookup(
    pairs: Dict[Tuple[str, str], str | float],
    a: str,
    b: str,
) -> str | float:
    if (a, b) in pairs:
        return pairs[(a, b)]
    if (b, a) in pairs:
        return pairs[(b, a)]
    return ""


def _resolve_metric_list(metrics: Iterable[str]) -> List[str]:
    metric_set = set(metrics)
    metric_list = [m for m in DEFAULT_METRICS if m in metric_set]
    if not metric_list:
        metric_list = list(DEFAULT_METRICS)
    return metric_list


def _parse_pair_label(pair: str) -> Tuple[str, str] | None:
    if not pair or " vs " not in pair:
        return None
    left, right = pair.split(" vs ", 1)
    return left.strip(), right.strip()


def _collect_effect_sizes(
    pair_label: str,
    metrics_payload: Dict[str, Dict[str, object]],
    metric_list: Iterable[str],
    data: Dict[str, Dict[Tuple[str, str], float]],
) -> None:
    parsed_pair = _parse_pair_label(pair_label)
    if not parsed_pair:
        return
    for metric in metric_list:
        metric_payload = metrics_payload.get(metric, {})
        effect_size = metric_payload.get("effect_size")
        if effect_size is None:
            raise ValueError(f"Missing effect_size for metric '{metric}' in pair '{pair_label}'.")
            continue
        try:
            value = float(effect_size)
        except (TypeError, ValueError):
            continue
        data.setdefault(metric, {})[parsed_pair] = value


def _load_dataset_effect_sizes(
    dataset_dir: Path,
    metric_list: Iterable[str],
    dataset_name: str
) -> Dict[str, Dict[Tuple[str, str], float]]:
    data: Dict[str, Dict[Tuple[str, str], float]] = {}
    json_file = f"pan_metrics_significance_{dataset_name}.json"
    if not (dataset_dir / json_file).exists():
        print(f"No {json_file} found in {dataset_dir}")

    with open(dataset_dir / json_file, "r") as f:
        payload = json.load(f)

    for pair_label, payload in payload.items():
        metrics_payload = payload.get("metrics", {})
        if isinstance(metrics_payload, dict):
            _collect_effect_sizes(
                pair_label, metrics_payload, metric_list, data
            )
    return data


def _format_effect_size(value: str | float) -> str:
    if value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number:.2f}"


def build_table(
    base_dir: Path,
    datasets: Iterable[str],
    metrics: Iterable[str],
    method_order: Iterable[str],
) -> str:
    method_list = list(method_order)
    metric_list = _resolve_metric_list(metrics)
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

    for dataset_name in datasets:
        if not base_dir.exists():
            print(f"[{dataset_name}]: No such dataset directory: {base_dir}")
            continue
        metrics_data = _load_dataset_metrics(base_dir, metric_list, dataset_name)
        if not metrics_data:
            continue

        lines.append(f"\\multicolumn{{{ncols}}}{{@{{}}c@{{}}}}{{\\emph{{{_dataset_display(dataset_name)}}}}} \\\\")

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


def build_effect_size_table(
    base_dir: Path,
    datasets: Iterable[str],
    metrics: Iterable[str],
    method_order: Iterable[str],
) -> str:
    method_list = list(method_order)
    metric_list = _resolve_metric_list(metrics)
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
        dataset_dir = base_dir
        if not dataset_dir.exists():
            continue
        metrics_data = _load_dataset_effect_sizes(dataset_dir, metric_list, dataset_name=dataset)
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
                    effect_size = _pair_lookup(pairs, method_row, method_col)
                    row_cells.append(_format_effect_size(effect_size))
            lines.append(f"  {_method_display(method_row)} & " + " & ".join(row_cells) + " \\\\")

        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}}")
    lines.append(
        "\\caption{Pairwise effect sizes (Cohens $d_z$) for each metric. "
        "Effect sizes are reported for the same corrected paired tests used in the significance table.}"
    )
    lines.append("\\label{tab:permutation-effect-sizes}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main() -> None:
    LOCAL_SAVE_PATH = (
        Path(__file__).resolve().parents[1] / CONFIG.SAVE_PATH / "reproduction/pan_metrics"
    )
    print(f"Local save path: {LOCAL_SAVE_PATH}")
    parser = argparse.ArgumentParser(
        description="Build a LaTeX table from extracted per-metric significance CSVs."
    )
    parser.add_argument(
        "--base-dir",
        default=LOCAL_SAVE_PATH / "statistical_significance/extracted_per_metric",
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
        datasets = list(CONFIG.DATASET_TRANSLATIONS.keys())

    latex = build_table(
        base_dir=base_dir,
        datasets=datasets,
        metrics=args.metrics,
        method_order=DEFAULT_METHOD_ORDER,
    )
    effect_size_latex = build_effect_size_table(
        base_dir=base_dir.parent,
        datasets=datasets,
        metrics=args.metrics,
        method_order=DEFAULT_METHOD_ORDER,
    )

    out_dir = base_dir.parent / "to_tex"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    with open(out_path, "w") as f:
        f.write(latex)
        print(f"Wrote LaTeX table to {out_path}")
    effect_size_out_path = out_dir / "effect_size_significance_table_all_metrics.tex"
    with open(effect_size_out_path, "w") as f:
        f.write(effect_size_latex)
        print(f"Wrote effect size LaTeX table to {effect_size_out_path}")


if __name__ == "__main__":
    main()
