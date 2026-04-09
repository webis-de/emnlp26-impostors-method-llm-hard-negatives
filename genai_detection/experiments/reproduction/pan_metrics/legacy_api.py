from __future__ import annotations

"""Legacy API surface for PAN metrics (compatibility with old imports)."""

from dataclasses import asdict
import json
import logging
from pathlib import Path

import numpy as np

from .pan_cv import PANEvaluator, SplitManager
from .pan_metric_computation import PANMetricComputer
from .pan_storage import PANMetricsRecord, PANMetricsStore

logger = logging.getLogger(__name__)


def get_pan_metrics(
    predictions,
    y_true,
    dataset_name: str,
    n_splits: int = 10,
    n_repeats: int = 5,
    random_state: int = 42,
    ci_level: float = 0.95,
    n_boot: int = 10000,
):
    """
    Compute PAN metrics for each method name and return a structured summary.

    Keeps the historical behavior of reusing MongoDB-stored metrics when available.
    """
    evaluator = PANEvaluator(
        PANMetricComputer(),
        SplitManager(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state),
    )
    store = PANMetricsStore()

    y_true_np = np.asarray(y_true)
    if y_true_np.ndim != 1:
        y_true_np = y_true_np.reshape(-1)
    n_samples = int(len(y_true_np))
    if n_samples == 0:
        raise ValueError("y_true must contain at least one sample.")

    normalized_predictions: dict[str, np.ndarray] = {}
    for method_name, method_scores in predictions.items():
        method_scores_np = np.asarray(method_scores, dtype=float)
        if method_scores_np.ndim != 1:
            method_scores_np = method_scores_np.reshape(-1)
        if len(method_scores_np) != n_samples:
            raise ValueError(
                f"{method_name}: expected {n_samples} scores, got {len(method_scores_np)}."
            )
        normalized_predictions[method_name] = method_scores_np

    precomputed_splits = evaluator.split_manager.build_splits(y_true_np)
    logger.info("Precomputed %d splits using %d samples.", len(precomputed_splits), n_samples)

    pan_metrics: dict[str, dict] = {}

    for method_name, method_scores in normalized_predictions.items():
        stored = store.get_record(dataset_name, method_name)
        if stored is not None:
            stored_n_samples = stored.get("n_samples")
            if stored_n_samples is not None and int(stored_n_samples) != n_samples:
                logger.warning(
                    "Stored PAN metrics for %s use n_samples=%s, but current data has n_samples=%d.",
                    method_name,
                    stored_n_samples,
                    n_samples,
                )
            stored = store.ensure_metric_values(stored, evaluator.extract_metric_values)
            pan_metrics[method_name] = _record_to_legacy_payload(stored)
            logger.info("Skipping %s (PAN metrics already stored).", method_name)
            continue

        cv_result = evaluator.compute_cv(
            y_true=y_true_np,
            scores=method_scores,
            ci_level=ci_level,
            n_boot=n_boot,
            splits=precomputed_splits,
        )

        pan_metrics[method_name] = asdict(cv_result)
        record = PANMetricsRecord(
            dataset_name=dataset_name,
            method_name=method_name,
            n_samples=n_samples,
            metrics_mean=cv_result.metrics_mean,
            metrics_std=cv_result.metrics_std,
            metrics_ci=cv_result.metrics_ci,
            pan_metrics_per_split=[asdict(result) for result in cv_result.per_split],
            split_config=cv_result.split_config,
        )
        store.save_record(record)
        logger.info("Obtained summary of PAN metrics for method %s.", method_name)

    return pan_metrics


def save_pan_metrics(pan_metrics, save_path: Path, dataset_name: str | None = None) -> Path:
    """
    Persist PAN metrics to JSON in the provided directory.
    """
    assert save_path.exists(), f"Directory {save_path} does not exist."
    serializable = _jsonify_value(pan_metrics)
    if dataset_name is None:
        save_file = save_path / "pan_metrics.json"
    else:
        save_file = save_path / f"{dataset_name}_pan_metrics.json"
    with open(save_file, "w") as f:
        json.dump(serializable, f, indent=2)
        logger.info("PAN metrics saved to %s.", save_file)
    return save_file




def _record_to_legacy_payload(record: dict) -> dict:
    return {
        "split_config": record.get("split_config"),
        "per_split": record.get("pan_metrics_per_split", []),
        "metric_values": record.get("metric_values", {}),
        "metrics_mean": record.get("metrics_mean", {}),
        "metrics_std": record.get("metrics_std", {}),
        "metrics_ci": record.get("metrics_ci", {}),
        "n_samples": record.get("n_samples"),
    }


def _jsonify_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonify_value(v) for k, v in value.items()}
    return value
