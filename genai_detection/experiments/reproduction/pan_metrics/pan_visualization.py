from __future__ import annotations

"""Visualization helpers for PAN metrics."""

import logging
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import PANMetricComputer
from genai_detection.experiments.reproduction.pan_metrics.pan_storage import PANMetricsStore

logger = logging.getLogger(__name__)

def _default_heatmap_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / CONFIG.SAVE_PATH / "reproduction" / "pan_metrics" / "heatmaps"


def _ordered_methods(methods: Sequence[str]|None) -> list[str]:
    if methods is None:
        methods = list(CONFIG.LABEL_TRANSLATIONS.keys())
    preferred = list(CONFIG.LABEL_TRANSLATIONS.keys())
    preferred_pos = {m: i for i, m in enumerate(preferred)}
    return sorted(methods, key=lambda m: (preferred_pos.get(m, 10_000), m))


def _valid_metric_names(names: Sequence[str]) -> list[str]:
    valid = set(CONFIG.SCORE_TRANSLATIONS.keys())
    filtered = [name for name in names if name in valid]
    ordered = [m for m in list(CONFIG.SCORE_TRANSLATIONS.keys()) if m in filtered]
    ordered += sorted([m for m in filtered if m not in ordered])
    return ordered


def _split_labels(n_splits: int, n_repeats: int | None) -> list[str]:
    if n_splits > 0 and n_repeats > 0:
        labels: list[str] = []
        for idx in range(n_repeats * n_splits):
            repeat = idx // n_splits + 1
            fold = idx % n_splits + 1
            labels.append(f"r{repeat:02d}-f{fold:02d}")
        return labels


def _load_pan_metrics_across_splits(
    *,
    store: PANMetricsStore,
    dataset_name: str,
    methods: Sequence[str],
    per_split_feature_name: str,
    n_folds: int | None = 10,
    n_repeats: int | None = 10,
) -> dict[str, dict[str, list[float]]]:
    """
    Load per-split PAN metrics grouped by method.
    """
    cursor = store.mongo.pan_metrics_collection.find(
        {"dataset_name": dataset_name, "method_name": {"$in": list(methods)}},
        {"_id": 0},
    )
    docs = list(cursor)
    n_samples_per_method : dict[str, dict[str, int]] = {}
    if docs:
        per_split_by_method: dict[str, dict[list[float]]] = {m: {} for m in methods}
        for doc in docs:
            method_name = doc.get("method_name")
            if method_name not in per_split_by_method:
                continue
            payload = doc.get(per_split_feature_name)
            split_config = doc.get("split_config")
            n_samples = doc.get("n_samples")

            if n_folds != split_config.get("n_splits") or n_repeats != split_config.get("n_repeats"):
                continue

            for metric in list(CONFIG.SCORE_TRANSLATIONS.keys()):
                if (method_name in n_samples_per_method and
                    n_samples_per_method[method_name].get(metric, 0) >= n_samples):
                    continue
                if method_name not in n_samples_per_method:
                    n_samples_per_method[method_name] = {}
                    per_split_by_method[method_name] = {}
                # payload is list of dicts, each dict is a split's metrics'
                per_split_by_method[method_name][metric] = [split.get(method_name, 0.0) for split in payload]
                n_samples_per_method[method_name][metric] = n_samples

        return per_split_by_method

    logger.warning(
        "No documents found in MongoDB collection '%s' for dataset=%s and methods=%s. Falling back to "
        "`pan_metrics.pan_metrics_per_split`.",
        per_split_feature_name,
        dataset_name,
        list(methods),
    )
    return {}


def plot_pan_metrics_heatmap_per_split(
    dataset_names: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    save_path: Path | None = None,
    *,
    store: PANMetricsStore | None = None,
    per_split_collection_name: str = "pan_metrics_per_split",
    n_folds: int | None = 10,
    n_repeats: int | None = 10,
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    """
    Create and save one heatmap figure per dataset.

    - Y axis: CV splits (folds × repeats).
    - X axis: methods (repeated once per metric score).
    - Blocks: one contiguous block per metric score (from `pan_metrics.metrics_mean` keys).

    Figures are saved as PDF and SVG to `results/reproduction/pan_metrics/heatmaps`.
    """
    store = store or PANMetricsStore()
    save_path = save_path or _default_heatmap_dir()
    save_path.mkdir(parents=True, exist_ok=True)

    if dataset_names is None:
        dataset_names = sorted(store.mongo.pan_metrics_collection.distinct("dataset_name"))
    if not dataset_names:
        raise ValueError("No datasets found in MongoDB collection 'pan_metrics'.")

    figures: dict[str, tuple[plt.Figure, plt.Axes]] = {}

    for dataset_name in dataset_names:
        methods_list = _ordered_methods(methods)

        pan_metrics_across_splits = _load_pan_metrics_across_splits(
            store=store,
            dataset_name=dataset_name,
            methods=methods_list,
            per_split_feature_name=per_split_collection_name,
            n_folds=n_folds,
            n_repeats=n_repeats,
        )

        split_labels = _split_labels(n_splits=n_folds, n_repeats=n_repeats)

        method_labels = [CONFIG.LABEL_TRANSLATIONS.get(m, m) for m in methods_list]
        metric_labels = []
        n_metrics = len(list(CONFIG.SCORE_TRANSLATIONS.keys()))

        xticklabels = method_labels * n_metrics

        matrix = np.full((n_repeats * n_folds, len(methods_list) * n_metrics), np.nan, dtype=float)
        for metric_idx, (metric, metric_label) in enumerate(CONFIG.SCORE_TRANSLATIONS.items()):
            metric_labels.append(metric_label)
            for method_idx, method_name in enumerate(methods_list):
                per_split_rows = pan_metrics_across_splits.get(method_name, {})
                col = metric_idx * len(methods_list) + method_idx
                for split_idx in range(min(n_repeats * n_folds, len(per_split_rows))):
                    if metric not in per_split_rows.keys():
                        raise ValueError(f"Metric {metric} not found in per_split_rows: {per_split_rows.keys()}")
                    raw = per_split_rows[metric][split_idx]
                    if raw is None:
                        continue
                    try:
                        matrix[split_idx, col] = float(raw)
                    except (TypeError, ValueError):
                        continue

        # Figure sizing: keep readable but avoid extreme layouts.
        fig_w = max(14.0, min(60.0, 0.22 * matrix.shape[1]))
        fig_h = max(8.0, min(30.0, 0.20 * matrix.shape[0]))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.set_theme(context="paper", style="white")
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            xticklabels=xticklabels,
            yticklabels=split_labels,
            cbar_kws={"label": "Score"},
            # annot=True,
        )

        # Reduce y tick density for large numbers of splits.
        max_yticks = 30
        if n_repeats * n_folds > max_yticks:
            step = int(math.ceil(n_repeats * n_folds / max_yticks))
            ax.set_yticks([i + 0.5 for i in range(0, n_repeats * n_folds, step)])
            ax.set_yticklabels([split_labels[i] for i in range(0, n_repeats * n_folds, step)], fontsize=12)
        else:
            ax.tick_params(axis="y", labelsize=12)

        ax.tick_params(axis="x", labelrotation=90, labelsize=12)
        ax.set_xlabel("Method", fontsize=15)
        ax.set_ylabel("Split", fontsize=15)

        dataset_label = CONFIG.DATASET_TRANSLATIONS.get(dataset_name, dataset_name)
        ax.set_title(f"PAN metrics per split - {dataset_label}", fontsize=16)

        # Draw vertical separators between score blocks.
        n_methods = len(methods_list)
        for i in range(1, len(metric_labels)):
            ax.axvline(i * n_methods, color="white", linewidth=3)
            ax.axvline(i * n_methods, color="black", linewidth=0.6, alpha=0.5)

        # Add a top axis with score labels centered per block.
        top = ax.secondary_xaxis("top")
        centers = [(i * n_methods) + (n_methods / 2) for i in range(len(metric_labels))]
        top.set_xticks(centers)
        top.set_xticklabels([CONFIG.SCORE_TRANSLATIONS.get(s, s) for s in metric_labels], fontsize=10)
        top.tick_params(axis="x", length=0)
        top.set_xlabel("Metric", fontsize=12)

        fig.tight_layout()

        safe_dataset = str(dataset_name).replace(" ", "_").replace("/", "_")
        base = f"pan_metrics_heatmap_per_split_{safe_dataset}"
        for fmt in ["pdf", "svg"]:
            out_path = save_path / f"{base}.{fmt}"
            fig.savefig(out_path, bbox_inches="tight")
            logger.info("Saved heatmap to %s", out_path)

        figures[dataset_name] = (fig, ax)

    return figures


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
