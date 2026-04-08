#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from genai_detection.config import CONFIG


def _iter_dataset_dirs(base_dir: Path, datasets: Iterable[str] | None) -> List[Path]:
    if datasets:
        return [base_dir / name for name in datasets]
    return [p for p in base_dir.iterdir() if p.is_dir()]


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


def extract_tables(base_dir: Path, datasets: Iterable[str] | None = None) -> None:
    dataset_dirs = _iter_dataset_dirs(base_dir, datasets)
    if not dataset_dirs:
        raise ValueError(f"No dataset directories found under {base_dir}")

    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")

        json_files = sorted(dataset_dir.glob("pan_metrics_significance_*.json"))
        if not json_files:
            continue

        entries = []
        methods = set()
        for path in json_files:
            with open(path, "r") as f:
                payload = json.load(f)
            pair = payload.get("pair")
            metrics = payload.get("metrics", {})
            if not pair or not metrics:
                continue
            left, right = _parse_pair(pair)
            methods.update([left, right])
            entries.append((left, right, metrics))

        if not entries:
            continue

        method_order = sorted(methods)
        method_index = {name: idx for idx, name in enumerate(method_order)}

        rows_by_metric: Dict[str, List[Tuple[str, str]]] = {}

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
            continue

        out_dir = dataset_dir / "extracted_per_metric"
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = dataset_dir.name
        for metric, rows in rows_by_metric.items():
            out_path = out_dir / f"significance_{dataset_name}_{metric}.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["pair", "min_significance_level"])
                for pair_label, min_level in sorted(rows):
                    writer.writerow([pair_label, min_level])


def main() -> None:
    LOCAL_SAVE_PATH = (
        Path(__file__).resolve().parents[1] / CONFIG.SAVE_PATH / "reproduction"
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
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset directory names to process (e.g., blog student).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    extract_tables(base_dir=base_dir, datasets=args.datasets)


if __name__ == "__main__":
    main()
