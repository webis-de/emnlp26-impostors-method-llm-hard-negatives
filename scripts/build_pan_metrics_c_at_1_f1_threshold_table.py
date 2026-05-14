#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[1] / CONFIG.SAVE_PATH / "reproduction"
)
OUTPUT_DIR = LOCAL_SAVE_PATH / "pan_metrics_tex"
OUTPUT_FILENAME = "table-impostor-method-reproduction-c-at-one-f1-thresholds.tex"


@dataclass(frozen=True)
class ThresholdRowSpec:
    method_name: str
    citation: str
    display_name: str


@dataclass(frozen=True)
class ThresholdSummary:
    dataset_name: str
    method_name: str
    n_samples: int
    lower_threshold: float
    lower_threshold_std: float
    upper_threshold: float
    upper_threshold_std: float
    f1_threshold: float | None
    f1_threshold_std: float | None
    n_splits_with_thresholds: int


DEFAULT_ROWS: Sequence[ThresholdRowSpec] = [
    ThresholdRowSpec(
        method_name="in_domain",
        citation=r"\cite{koppel_determining_2014}",
        display_name="Koppel & Winter, 2014",
    ),
    ThresholdRowSpec(
        method_name="asgalf",
        citation=r"\cite{khonji_slightly_modified_2014}",
        display_name="Khonji & Iraqi, 2014",
    ),
    ThresholdRowSpec(
        method_name="homotopy",
        citation=r"\cite{gutierrez2015homotopy}",
        display_name="Gutierrez et al., 2015",
    ),
    ThresholdRowSpec(
        method_name="std_impostor",
        citation=r"\cite{kestemont_authenticating_2016}",
        display_name="Kestemont et al., 2016",
    ),
    ThresholdRowSpec(
        method_name="potha2017",
        citation=r"\cite{potha_improved_2017}",
        display_name="Potha et al., 2017",
    ),
    ThresholdRowSpec(
        method_name="bdi",
        citation=r"\cite{nagy_bootstrap_2024}",
        display_name="Nagy, 2024",
    ),
]

DEFAULT_DATASETS: Sequence[str] = [CONFIG.STUDENT_ESSAYS, CONFIG.BLOG]


def _dataset_display(dataset_name: str) -> str:
    return CONFIG.DATASET_TRANSLATIONS.get(dataset_name, dataset_name)


def _n_samples_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _matches_split_config(
    doc: Mapping[str, Any],
    *,
    n_splits: int | None,
    n_repeats: int | None,
) -> bool:
    split_config = doc.get("split_config") or {}
    if n_splits is not None and _int_or_none(split_config.get("n_splits")) != n_splits:
        return False
    if (
        n_repeats is not None
        and _int_or_none(split_config.get("n_repeats")) != n_repeats
    ):
        return False
    return True


def _normalize_pan_metrics_doc(doc: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize current and legacy PAN metrics records into the stored canonical shape.
    """
    doc_dict = dict(doc)
    pan_payload = doc_dict.get("pan_metrics", {}) if isinstance(doc_dict, dict) else {}

    if not isinstance(pan_payload, Mapping):
        pan_payload = {}

    return {
        "dataset_name": doc_dict.get("dataset_name") or pan_payload.get("dataset_name"),
        "method_name": doc_dict.get("method_name") or pan_payload.get("method_name"),
        "n_samples": doc_dict.get("n_samples") or pan_payload.get("n_samples"),
        "metrics_mean": doc_dict.get("metrics_mean") or pan_payload.get("metrics_mean") or {},
        "metrics_std": doc_dict.get("metrics_std") or pan_payload.get("metrics_std") or {},
        "metrics_ci": doc_dict.get("metrics_ci") or pan_payload.get("metrics_ci") or {},
        "pan_metrics_per_split": doc_dict.get("pan_metrics_per_split")
        or doc_dict.get("per_split")
        or pan_payload.get("per_split")
        or pan_payload.get("pan_metrics_per_split")
        or [],
        "split_config": pan_payload.get("split_config") or doc_dict.get("split_config"),
        "metric_values": pan_payload.get("metric_values"),
    }


def _threshold_values(doc: Mapping[str, Any]) -> list[tuple[float, float, float | None]]:
    per_split = doc.get("pan_metrics_per_split") or doc.get("per_split") or []
    values: list[tuple[float, float, float | None]] = []
    for split_result in per_split:
        if not isinstance(split_result, Mapping):
            continue
        lower_threshold = _float_or_none(split_result.get("lower_threshold"))
        upper_threshold = _float_or_none(split_result.get("upper_threshold"))
        if lower_threshold is None or upper_threshold is None:
            continue
        f1_threshold = _float_or_none(split_result.get("f1_threshold"))
        values.append((lower_threshold, upper_threshold, f1_threshold))
    print(values)
    return values


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("values must contain at least one threshold.")
    return float(mean(values)), float(stdev(values)) if len(values) > 1 else 0.0


def _summarize_doc(
    doc: Mapping[str, Any],
) -> ThresholdSummary | None:
    threshold_values = _threshold_values(doc)
    if not threshold_values:
        return None
    lower_thresholds = [lower for lower, _, _ in threshold_values]
    upper_thresholds = [upper for _, upper, _ in threshold_values]
    f1_thresholds = [
        f1_threshold
        for _, _, f1_threshold in threshold_values
        if f1_threshold is not None
    ]
    lower_threshold, lower_threshold_std = _mean_std(lower_thresholds)
    upper_threshold, upper_threshold_std = _mean_std(upper_thresholds)
    f1_threshold, f1_threshold_std = (
        _mean_std(f1_thresholds) if f1_thresholds else (None, None)
    )
    return ThresholdSummary(
        dataset_name=str(doc.get("dataset_name")),
        method_name=str(doc.get("method_name")),
        n_samples=_n_samples_value(doc.get("n_samples")),
        lower_threshold=lower_threshold,
        lower_threshold_std=lower_threshold_std,
        upper_threshold=upper_threshold,
        upper_threshold_std=upper_threshold_std,
        f1_threshold=f1_threshold,
        f1_threshold_std=f1_threshold_std,
        n_splits_with_thresholds=len(threshold_values),
    )


def select_threshold_summary(
    docs: Iterable[Mapping[str, Any]],
    *,
    aggregation: str | None = None,
    n_splits: int | None = None,
    n_repeats: int | None = None,
) -> ThresholdSummary | None:
    """
    Select the MongoDB record with the largest n_samples and summarize thresholds.
    """
    best_summary: ThresholdSummary | None = None
    for raw_doc in docs:
        normalized = _normalize_pan_metrics_doc(raw_doc)
        if not _matches_split_config(
            normalized,
            n_splits=n_splits,
            n_repeats=n_repeats,
        ):
            continue
        summary = _summarize_doc(normalized)
        if summary is None:
            continue
        if best_summary is None or summary.n_samples > best_summary.n_samples:
            best_summary = summary
    return best_summary


def _load_threshold_summary(
    mongo: ParaphraseMongoDB,
    *,
    dataset_name: str,
    method_name: str,
    aggregation: str | None,
    n_splits: int | None,
    n_repeats: int | None,
) -> ThresholdSummary | None:
    docs = list(
        mongo.pan_metrics_collection.find(
            {"dataset_name": dataset_name, "method_name": method_name},
            {"_id": 0},
        ).sort([("n_samples", -1), ("_id", -1)])
    )
    assert docs, f"No records found for dataset={dataset_name} method={method_name}."
    # TODO: wait until mongodb colelction is filled again
    summary = select_threshold_summary(
        docs,
        aggregation=aggregation,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )
    if summary is None:
        logger.warning(
            "No c@1 thresholds found for dataset=%s method=%s.",
            dataset_name,
            method_name,
        )
    else:
        logger.info(
            "Using %s/%s record with %d samples and %d threshold splits.",
            dataset_name,
            method_name,
            summary.n_samples,
            summary.n_splits_with_thresholds,
        )
    return summary


def collect_thresholds(
    mongo: ParaphraseMongoDB,
    *,
    datasets: Iterable[str],
    rows: Sequence[ThresholdRowSpec],
    aggregation: str | None = None,
    n_splits: int | None = None,
    n_repeats: int | None = None,
) -> dict[tuple[str, str], ThresholdSummary]:
    thresholds: dict[tuple[str, str], ThresholdSummary] = {}
    for dataset_name in datasets:
        for row in rows:
            summary = _load_threshold_summary(
                mongo,
                dataset_name=dataset_name,
                method_name=row.method_name,
                aggregation=aggregation,
                n_splits=n_splits,
                n_repeats=n_repeats,
            )
            if summary is not None:
                thresholds[(dataset_name, row.method_name)] = summary
    return thresholds


def _format_threshold(
    value: float | None,
    std: float | None,
    decimals: int,
) -> str:
    if value is None or std is None:
        return "--"
    return rf"${value:.{decimals}f} \pm {std:.{decimals}f}$"


def _format_n_samples_comment(summary: ThresholdSummary | None) -> str:
    if summary is None or summary.n_samples < 0:
        return "% n_sample=--"
    return f"% n_sample={summary.n_samples}"


def build_latex_table(
    thresholds: Mapping[tuple[str, str], ThresholdSummary],
    *,
    datasets: Sequence[str],
    rows: Sequence[ThresholdRowSpec],
    aggregation: str | None = None,
    decimals: int = 2,
) -> str:
    lines: list[str] = []
    lines.append("% Auto-generated; do not edit by hand.")
    lines.append(
        "% Threshold values are mean \\pm sample standard deviation over per-split "
        "optimized thresholds from the selected MongoDB record."
    )
    for row in rows:
        lines.append(f'% "{row.method_name}": "{row.display_name}",')
    lines.append("")
    lines.append(r"\begin{table}[t]\small")
    lines.append(r"  \tabcolsep=0.11cm")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"  \toprule")
    lines.append(
        r"\textbf{Author} & \textbf{Dataset} & \textbf{$\tau_{low}$} & \textbf{$\tau_{high}$} & \textbf{$\tau_{F1}$} \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        for dataset_idx, dataset_name in enumerate(datasets):
            summary = thresholds.get((dataset_name, row.method_name))
            citation = row.citation if dataset_idx == 0 else ""
            lower_threshold = _format_threshold(
                summary.lower_threshold if summary else None,
                summary.lower_threshold_std if summary else None,
                decimals,
            )
            upper_threshold = _format_threshold(
                summary.upper_threshold if summary else None,
                summary.upper_threshold_std if summary else None,
                decimals,
            )
            f1_threshold = _format_threshold(
                summary.f1_threshold if summary else None,
                summary.f1_threshold_std if summary else None,
                decimals,
            )
            n_samples_comment = _format_n_samples_comment(summary)
            lines.append(
                f"{citation} & {_dataset_display(dataset_name)} & "
                f"{lower_threshold} & {upper_threshold} & {f1_threshold} \\\\ {n_samples_comment}"
            )

    lines.append(r" \bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{Average thresholds optimized on $10 \times 20$ fold CV for different extensions of the original "
        r"Impostors Method. The lower and upper thresholds define the c@1 rejection band, "
        r"whereas $\tau_{F1}$ is the threshold optimized for F1, precision, recall and accuracy. Values are "
        r"reported as mean $\pm$ standard deviation over the per-fold thresholds from cross-validation. These dataset-specific thresholds were used to obtain the scores reported in "
        r"Table~\ref{table-impostor-method-reproduction-results}.}"
    )
    lines.append(r"\label{tab:ablation_thresholds}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LaTeX table of optimized PAN metric thresholds from MongoDB."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / OUTPUT_FILENAME,
        help="Output .tex path.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("mean", "median"),
        default="mean",
        help="Deprecated; ignored. Thresholds are reported as mean +/- standard deviation.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimal places to show in the table.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Only use records with this split_config.n_splits value.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Only use records with this split_config.n_repeats value.",
    )
    parser.add_argument(
        "--remote-ray",
        action="store_true",
        help="Use the remote Ray MongoDB connection instead of the local forwarded MongoDB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mongo = ParaphraseMongoDB(local_ray=not args.remote_ray)
    thresholds = collect_thresholds(
        mongo,
        datasets=DEFAULT_DATASETS,
        rows=DEFAULT_ROWS,
        aggregation=args.aggregation,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
    )
    latex = build_latex_table(
        thresholds,
        datasets=DEFAULT_DATASETS,
        rows=DEFAULT_ROWS,
        aggregation=args.aggregation,
        decimals=args.decimals,
    )

    args.output.write_text(latex + "\n", encoding="utf-8")
    print(f"Wrote LaTeX table to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
