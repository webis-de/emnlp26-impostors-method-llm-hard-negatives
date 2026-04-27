#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from genai_detection.config import CONFIG


def _parse_pair(pair: str) -> Tuple[str, str]:
    if " vs " not in pair:
        raise ValueError(f"Unexpected pair format: '{pair}'")
    left, right = pair.split(" vs ", 1)
    return left.strip(), right.strip()


def _min_significance_level(significant: Dict[str, bool]) -> str:
    levels = sorted((float(k) for k in significant.keys()))
    for level in levels:
        if significant.get(str(level), False):
            return f"{level:g}"
    return "ns"


def _canonical_pair(
    pair: Tuple[str, str],
    method_index: Dict[str, int],
) -> Tuple[str, str]:
    left, right = pair
    if method_index[left] <= method_index[right]:
        return left, right
    return right, left


def extract_tables(base_dir: Path) -> None:
    if not base_dir or not base_dir.exists():
        raise ValueError(f"No dataset directory '{base_dir}' found")

    json_files = sorted(base_dir.glob("pan_metrics_significance_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No pan_metrics_significance_*.json files found in {base_dir}")

    out_dir = base_dir / "extracted_per_metric"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = {}    # keys: dataset_name, values: (left, right, metrics)
    methods = {}    # keys: dataset_name, values: set of method names
    for path in json_files:
        dataset_name = "_".join(path.stem.split("_")[3:])
        with open(path, "r") as f:
            payload = json.load(f)
            print(payload.keys())
        for pair in list(payload.keys()):
            values = payload.get(pair, {})
            left, right = _parse_pair(pair)
            metrics = values.get("metrics", {})
            methods.setdefault(dataset_name, set()).update([left, right])
            entries.setdefault(dataset_name, []).append((left, right, metrics))
        method_order = sorted(methods[dataset_name])

    if not entries:
        raise ValueError(f"No significant entries found in {base_dir}")

    method_index = {name: idx for idx, name in enumerate(method_order)}

    rows_by_metric: Dict[str, List[Tuple[str, str]]] = {}

    for dataset_name, entries in entries.items():
        entries = sorted(entries, key=lambda x: method_index[x[0]])
        for left, right, metrics in entries:
            canon_left, canon_right = _canonical_pair((left, right), method_index)
            # only keep upper diagonal comparisons
            if method_index[canon_left] >= method_index[canon_right]:
                continue
            pair_label = f"{canon_left} vs {canon_right}"
            for metric, metric_payload in metrics.items():
                significant = metric_payload.get("significant", {})
                min_level = _min_significance_level(significant) if significant else "ns"
                rows_by_metric.setdefault(metric, []).append((pair_label, min_level))

        if not rows_by_metric:
            raise ValueError(f"No significant rows found in {base_dir} for any metric")

        print(f"save for {dataset_name} to {out_dir}")
        for metric, rows in rows_by_metric.items():
            out_path = out_dir / f"significance_{dataset_name}_{metric}.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["pair", "min_significance_level"])
                for pair_label, min_level in sorted(rows):
                    writer.writerow([pair_label, min_level])
        print(f"Extracted significance tables for {dataset_name} to {out_dir}")


def main() -> None:
    LOCAL_SAVE_PATH = (
        Path(__file__).resolve().parents[1] / CONFIG.SAVE_PATH / "reproduction/pan_metrics"
    )
    print(f"Local save path: {LOCAL_SAVE_PATH}")
    parser = argparse.ArgumentParser(
        description="Extract per-metric significance tables from pan_metrics_significance JSON files."
    )
    parser.add_argument(
        "--base-dir",
        default=LOCAL_SAVE_PATH / "statistical_significance",
        help="Base directory containing dataset subdirectories.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    extract_tables(base_dir=base_dir)


if __name__ == "__main__":
    main()
