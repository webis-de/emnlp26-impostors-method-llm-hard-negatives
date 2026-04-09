from __future__ import annotations

"""Visualization helpers for PAN metrics."""

import logging
from pathlib import Path
from typing import Sequence

from matplotlib import pyplot as plt

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import PANMetricComputer
from genai_detection.experiments.reproduction.pan_metrics.pan_storage import PANMetricsStore

logger = logging.getLogger(__name__)

def plot_pan_metrics_boxplots(
    dataset_name: str | None = None,
    methods: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    save_path: Path | None = None,
    title: str | None = None,
    *,
    store: PANMetricsStore | None = None,
):
    """
    Plot per-fold PAN metrics as grouped boxplots (grouped by metric, colored by method).
    """
    store = store or PANMetricsStore()

    query: dict = {}
    if dataset_name is not None:
        query["dataset_name"] = dataset_name
    if methods is not None:
        query["method_name"] = {"$in": list(methods)}

    cursor = store.mongo.pan_metrics_collection.find(query, {"_id": 0})

    metrics_by_method: dict[str, dict[str, list[float]]] = {}
    split_config: dict[str, float | int] | None = None
    for doc in cursor:
        normalized = store._normalize_doc(doc)
        method_name = normalized.get("method_name")
        if not method_name:
            continue
        normalized = store.ensure_metric_values(
            normalized,
            PANMetricComputer.extract_metric_values,
        )
        metric_values = normalized.get("metric_values") or {}
        if metric_values:
            metrics_by_method[method_name] = metric_values
            if split_config is None:
                split_config = normalized.get("split_config")

    if not metrics_by_method:
        raise ValueError("No PAN metrics found for the given query.")

    if methods is None:
        methods_list = sorted(metrics_by_method.keys())
    else:
        methods_list = [method for method in methods if method in metrics_by_method]
        missing = [method for method in methods if method not in metrics_by_method]
        if missing:
            logger.warning("Missing PAN metrics for methods: %s", ", ".join(missing))

    if metrics is None:
        available = {m for vals in metrics_by_method.values() for m in vals.keys()}
        metrics_list = [m for m in list(CONFIG.SCORE_TRANSLATIONS.keys()) if m in available]
        metrics_list += sorted(m for m in available if m not in metrics_list)
    else:
        metrics_list = list(metrics)

    n_methods = len(methods_list)
    n_metrics = len(metrics_list)

    box_data: list[list[float]] = []
    box_positions: list[float] = []
    box_method_idx: list[int] = []

    for metric_idx, metric in enumerate(metrics_list):
        group_start = metric_idx * (n_methods + 1) + 1
        for method_idx, method in enumerate(methods_list):
            values = metrics_by_method.get(method, {}).get(metric, [])
            if not values:
                continue
            box_positions.append(group_start + method_idx)
            box_data.append(values)
            box_method_idx.append(method_idx)

    if not box_data:
        raise ValueError("No metric values found for plotting.")

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 2.5), 6))
    boxplot = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )

    colors = [
        CONFIG.LABEL_COLORS.get(method, "#4d4d4d")
        for method in methods_list
    ]
    for patch, method_idx in zip(boxplot["boxes"], box_method_idx):
        patch.set_facecolor(colors[method_idx])
        patch.set_edgecolor("black")

    group_centers = []
    for metric_idx in range(n_metrics):
        group_start = metric_idx * (n_methods + 1) + 1
        center = group_start + (n_methods - 1) / 2
        group_centers.append(center)

    metric_labels = [CONFIG.SCORE_TRANSLATIONS.get(m, m) for m in metrics_list]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(metric_labels, rotation=0)
    ax.set_xlabel("Metric", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)

    if title is None:
        n_splits = None
        n_repeats = None
        if split_config and "n_splits" in split_config and "n_repeats" in split_config:
            n_splits = split_config.get("n_splits")
            n_repeats = split_config.get("n_repeats")
            title = f"PAN Metrics ({int(n_splits)} Folds, {int(n_repeats)} Repetitions)"
        else:
            title = f"PAN Metrics"
        if dataset_name is not None:
            dataset_label = CONFIG.DATASET_TRANSLATIONS.get(dataset_name, dataset_name)
            title = f"{title} - {dataset_label}"
    ax.set_title(title, fontsize=16)

    legend_handles = []
    for i, method in enumerate(methods_list):
        label = CONFIG.LABEL_TRANSLATIONS.get(method, method)
        legend_handles.append(
            plt.Line2D([0], [0], color=colors[i], lw=6, label=label)
        )
    ax.legend(
        handles=legend_handles,
        title="Technique",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=13,
        title_fontsize=13,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=(0, 0, 1, 1))

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        suffix = dataset_name if dataset_name is not None else "all"
        for format in ["svg", "pdf"]:
            out_path = save_path / f"pan_metrics_boxplot_{suffix}.{format}"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            logger.info("Saved boxplot to %s", out_path)

    return fig, ax
