from __future__ import annotations
from dataclasses import dataclass
import json
import os
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_consistent_length

from genai_detection.config import CONFIG
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Metrics for a single evaluation using a fixed (lower, upper) threshold pair.
    """
    lower_threshold: float
    upper_threshold: float
    f1_threshold: float | None
    n_answered_c_at_1: int
    n_unanswered_c_at_1: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    c_at_1: float
    auroc: float
    auroc_c_at_1: float


@dataclass
class EvaluationCVResult:
    """
    Cross-validation summary for threshold tuning.

    - split_config: parameters used to generate CV splits.
    - per_split: list of EvaluationResult objects, one per fold.
    - metric_values: raw per-fold metric values (length = n_splits * n_repeats).
    - metrics_mean/std: summary statistics across folds.
    - metrics_ci: percentile bootstrap CI for the mean of each metric.
    - candidate_thresholds: candidate threshold grid, not the chosen lower/upper thresholds.

    Example values: split_config={"n_splits": 10, "n_repeats": 5, "test_size": 0.1, "random_state": 42},
    candidate_thresholds=[0.2, 0.5, 0.8], per_split=[EvaluationResult(...), ...],
    metrics_mean={"c_at_1": 0.72}, metrics_std={"c_at_1": 0.05},
    metric_values={"c_at_1": [0.7, 0.75]}, metrics_ci={"c_at_1": {"low": 0.65, "high": 0.78, "level": 0.95, "method": "bootstrap_percentile", "n_boot": 10000}}.
    """
    split_config: dict[str, float | int]
    candidate_thresholds: list[float] | None
    per_split: list[EvaluationResult]
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    metric_values: dict[str, list[float]]
    metrics_ci: dict[str, dict[str, float | int | str]]


class BinaryVerificationEvaluator:
    """
    Evaluate binary verification scores with threshold-dependent metrics,
    c@1, AUROC, and AUROC*c@1. Supports tuning a rejection band defined
    by lower/upper thresholds and summarizing performance across
    repeated stratified k-fold splits.

    Assumptions:
    - y_true contains binary gold labels: 0 or 1
    - scores are continuous verification scores
    - higher score => more evidence for class 1

    Prediction rule:
    - score > upper_threshold -> predict 1
    - score < lower_threshold -> predict 0
    - otherwise               -> unanswered (0.5)

    Notes:
    - AUROC is computed on raw scores and does not use a threshold.
    - Threshold tuning should be done on a validation set.
    """

    UNANSWERED = 0.5

    def __init__(self, positive_label: int = 1) -> None:
        self.positive_label = positive_label

    @staticmethod
    def _to_numpy(x: Sequence[float] | Sequence[int] | np.ndarray) -> np.ndarray:
        return np.asarray(x)

    def _validate_binary_inputs(
        self,
        y_true: Sequence[int] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ensure inputs are aligned, finite, and binary labels.
        Example values: y_true=[0, 1, 0], scores=[0.12, 0.87, 0.33].
        """
        y_true_np = self._to_numpy(y_true).astype(int)
        scores_np = self._to_numpy(scores).astype(float)

        check_consistent_length(y_true_np, scores_np)

        if np.isnan(scores_np).any() or np.isinf(scores_np).any():
            raise ValueError("scores must not contain NaN or infinite values.")

        y_target_type = type_of_target(y_true_np)
        if y_target_type != "binary":
            raise ValueError(
                f"y_true must be binary labels, got target type '{y_target_type}'."
            )

        return y_true_np, scores_np

    @staticmethod
    def _normalize_thresholds(
        scores: np.ndarray,
        thresholds: Iterable[float] | None,
    ) -> np.ndarray:
        """
        Return a sorted, unique array of candidate thresholds.
        If thresholds is None, use the observed scores plus two sentinels.
        Example values: scores=[0.1, 0.4, 0.9], thresholds=[0.2, 0.5, 0.8] or None.
        """
        if thresholds is None:
            unique_scores = np.unique(scores)
            return np.concatenate(
                (
                    [float(np.min(scores)) - 1e-12],
                    unique_scores,
                    [float(np.max(scores)) + 1e-12],
                )
            )

        threshold_array = np.asarray(list(thresholds), dtype=float)
        if threshold_array.size == 0:
            raise ValueError("thresholds must contain at least one value.")
        if np.isnan(threshold_array).any() or np.isinf(threshold_array).any():
            raise ValueError("thresholds must not contain NaN or infinite values.")
        return np.unique(threshold_array)

    @staticmethod
    def _summarize_metrics(
        results: Sequence[EvaluationResult],
        ci_level: float,
        n_boot: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]]]:
        """
        Compute per-metric mean/std across folds and percentile bootstrap CIs
        for the mean. For each metric, bootstrap draws resample the per-fold
        values with replacement, using the same list length
        (n_splits * n_repeats), repeated `n_boot` times. The CI bounds are
        the percentile cutoffs of those bootstrap means.
        Returns mean, std, raw values, and CI dicts.
        Example values: results=[EvaluationResult(...), ...], ci_level=0.95, n_boot=10000.
        """
        metric_values: dict[str, list[float]] = {
            "f1_threshold": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "accuracy": [],
            "c_at_1": [],
            "auroc": [],
            "auroc_c_at_1": [],
            "n_answered_c_at_1": [],
            "n_unanswered_c_at_1": [],
        }
        for result in results:
            metric_values["f1_threshold"].append(
                float(result.f1_threshold) if result.f1_threshold is not None else float("nan")
            )
            metric_values["precision"].append(result.precision)
            metric_values["recall"].append(result.recall)
            metric_values["f1"].append(result.f1)
            metric_values["accuracy"].append(result.accuracy)
            metric_values["c_at_1"].append(result.c_at_1)
            metric_values["auroc"].append(result.auroc)
            metric_values["auroc_c_at_1"].append(result.auroc_c_at_1)
            metric_values["n_answered_c_at_1"].append(float(result.n_answered_c_at_1))
            metric_values["n_unanswered_c_at_1"].append(float(result.n_unanswered_c_at_1))

        metrics_mean: dict[str, float] = {}
        metrics_std: dict[str, float] = {}
        for metric, values in metric_values.items():
            metrics_mean[metric] = float(np.mean(values))
            metrics_std[metric] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        metrics_ci: dict[str, dict[str, float | int | str]] = {}
        for metric, values in metric_values.items():
            arr = np.asarray(values, dtype=float)
            if len(arr) < 2 or n_boot <= 0:
                metrics_ci[metric] = {
                    "low": float(np.nan),
                    "high": float(np.nan),
                    "level": float(ci_level),
                    "method": "bootstrap_percentile",
                    "n_boot": int(n_boot),
                }
                continue
            boot = bootstrap(
                (arr,),
                np.mean,
                confidence_level=ci_level,
                n_resamples=n_boot,
                method="percentile",
                random_state=42,
            )
            low = float(boot.confidence_interval.low)
            high = float(boot.confidence_interval.high)
            metrics_ci[metric] = {
                "low": low,
                "high": high,
                "level": float(ci_level),
                "method": "bootstrap_percentile",
                "n_boot": int(n_boot),
            }

        return metrics_mean, metrics_std, metric_values, metrics_ci

    @classmethod
    def compute_c_at_1(
        cls,
        y_true: np.ndarray,
        y_preds: np.ndarray,
    ) -> float:
        """
        Compute c@1 as defined in PAN-style evaluation.
        Peñas & Rodrigo (2011): "A Simple Measure to Assess Non-response"

        `y_preds` must contain:
        - 0 for negative prediction
        - 1 for positive prediction
        - 0.5 or cls.UNANSWERED for unanswered
        Example values: y_true=[0, 1, 0], y_pred_with_reject=[0, 0.5, 1].
        """
        n = len(y_true)
        if n == 0:
            raise ValueError("Empty inputs are not allowed.")

        answered_mask = y_preds != cls.UNANSWERED
        unanswered_mask = ~answered_mask

        n_u = int(np.sum(unanswered_mask))
        n_c = int(np.sum(y_true[answered_mask] == y_preds[answered_mask]))

        return (1.0 / n) * (n_c + n_u * (n_c / n))

    @classmethod
    def _answered_only(
        cls,
        y_true: np.ndarray,
        y_pred_with_reject: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = y_pred_with_reject != cls.UNANSWERED
        return y_true[mask], y_pred_with_reject[mask].astype(int)

    def _optimize_f1_threshold(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        thresholds: Iterable[float] | None,
    ) -> tuple[float, float]:
        best_threshold = float(thresholds[0])
        best_f1 = -np.inf
        best_recall = -np.inf
        best_precision = -np.inf

        for threshold in thresholds:
            preds = (scores > threshold).astype(int)
            precision = precision_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)

            if f1 > best_f1 or (
                np.isclose(f1, best_f1)
                and (recall > best_recall or (np.isclose(recall, best_recall) and precision > best_precision))
            ):
                best_f1 = float(f1)
                best_recall = float(recall)
                best_precision = float(precision)
                best_threshold = float(threshold)

        return best_threshold, best_f1

    def evaluate_at_threshold(
        self,
        y_true: Sequence[int] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        lower_threshold: float,
        upper_threshold: float,
        f1_threshold: float | None = None,
    ) -> EvaluationResult:
        """
        Evaluate metrics for a fixed lower/upper threshold band.

        Scores in (lower_threshold, upper_threshold) are treated as unanswered (0.5).
        Example values: y_true=[0, 1, 0], scores=[0.2, 0.8, 0.4],
        lower_threshold=0.3, upper_threshold=0.7.

        If f1_threshold is provided, precision/recall/F1/accuracy are computed
        on all samples using that threshold (no rejection band).
        """

        y_true_np, scores_np = self._validate_binary_inputs(y_true=y_true, scores=scores)

        if len(np.unique(y_true_np)) < 2:
            raise ValueError("AUROC requires both classes to be present in y_true.")

        if lower_threshold > upper_threshold:
            raise ValueError("lower_threshold must be <= upper_threshold.")

        preds = np.full(shape=len(scores_np), fill_value=self.UNANSWERED, dtype=float)
        preds[scores_np > upper_threshold] = 1.0
        preds[scores_np < lower_threshold] = 0.0

        y_true_answered, y_pred_answered = self._answered_only(
            y_true_np,
            preds,
        )

        n_answered_c_at_1 = len(y_true_answered)
        n_unanswered_c_at_1 = len(y_true_np) - n_answered_c_at_1
        if f1_threshold is None:
            # happens if only c@1 values are computed during tuning
            if n_answered_c_at_1 == 0:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                accuracy = 0.0
            else:
                # fallback: Compute precision and recall only on answered problems
                precision = precision_score(
                    y_true_answered,
                    y_pred_answered,
                    zero_division=0,
                )
                recall = recall_score(
                    y_true_answered,
                    y_pred_answered,
                    zero_division=0,
                )
                # PAN: does not take into account non-answered:
                # cf. https://ceur-ws.org/Vol-2936/paper-147.pdf (2021)
                # cf. https://ceur-ws.org/Vol-3497/paper-199.pdf (2023)
                f1 = f1_score(
                    y_true_answered,
                    y_pred_answered,
                    zero_division=0,
                )

                # we compute accuracy only on answered problems, because otherwise we need to optimize a third threshold
                accuracy = accuracy_score(y_true_answered, y_pred_answered)
        else:
            preds_f1 = (scores_np > float(f1_threshold)).astype(int)
            precision = precision_score(y_true_np, preds_f1, zero_division=0)
            recall = recall_score(y_true_np, preds_f1, zero_division=0)
            f1 = f1_score(y_true_np, preds_f1, zero_division=0)
            accuracy = accuracy_score(y_true_np, preds_f1)

        c_at_1 = self.compute_c_at_1(y_true=y_true_np, y_preds=preds)
        auroc = roc_auc_score(y_true_np, scores_np)
        auroc_c_at_1 = auroc * c_at_1

        return EvaluationResult(
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            f1_threshold=None if f1_threshold is None else float(f1_threshold),
            n_answered_c_at_1=n_answered_c_at_1,
            n_unanswered_c_at_1=n_unanswered_c_at_1,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            c_at_1=float(c_at_1),
            auroc=float(auroc),
            auroc_c_at_1=float(auroc_c_at_1),
        )

    def tune_thresholds(
        self,
        y_true: Sequence[int] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        thresholds: Iterable[float] | None = None,
        n_splits: int = 10,
        n_repeats: int = 5,
        random_state: int = 42,
        ci_level: float = 0.95,
        n_boot: int = 10000,
    ) -> EvaluationCVResult:
        """
        Tune thresholds on repeated stratified k-fold splits and summarize
        performance on held-out folds.
        Optimize two thresholds for c@1, optimize a single threshold for F1 (used later for all other metrics).

        Threshold pairs are drawn from the candidate `thresholds` list and
        evaluated under the constraint `lower_threshold <= upper_threshold`.
        Scores between the thresholds are treated as unanswered.

        Returns per-split results plus mean/std and percentile bootstrap CIs.
        Example values: y_true=[0, 1, 0, 1], scores=[0.1, 0.9, 0.4, 0.7],
        thresholds=[0.2, 0.5, 0.8], n_splits=10, n_repeats=5, random_state=42, ci_level=0.95, n_boot=10000.
        """
        y_true_np, scores_np = self._validate_binary_inputs(y_true=y_true, scores=scores)

        if len(np.unique(y_true_np)) < 2:
            raise ValueError("Threshold tuning requires both classes to be present in y_true.")

        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 for cross-validation.")
        if n_repeats < 1:
            raise ValueError("n_repeats must be >= 1 for cross-validation.")
        if not 0.0 < ci_level < 1.0:
            raise ValueError("ci_level must be between 0 and 1.")
        if n_boot < 1:
            raise ValueError("n_boot must be >= 1.")

        split_config = {
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "test_size": float(1.0 / n_splits),
            "random_state": int(random_state),
        }

        per_split_results: list[EvaluationResult] = []

        try:
            splitter = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )
            split_iter = splitter.split(scores_np, y_true_np)
            logger.info(f"Splitting data with {split_config}.")
        except ValueError as exc:
            raise ValueError(
                "Stratified split failed. Ensure each class in y_true has at least two samples."
            ) from exc

        thresholds_list = (
            None if thresholds is None else [float(x) for x in thresholds]
        )

        try:
            for train_idx, test_idx in split_iter:
                y_true_train = y_true_np[train_idx]
                scores_train = scores_np[train_idx]
                y_true_test = y_true_np[test_idx]
                scores_test = scores_np[test_idx]

                threshold_values = self._normalize_thresholds(
                    scores=scores_train, thresholds=thresholds
                )

                # one F1 threshold per split/fold (used later for all metrics but c@1)
                best_f1_threshold, _ = self._optimize_f1_threshold(
                    y_true=y_true_train,
                    scores=scores_train,
                    thresholds=thresholds,
                )

                # optimize two thresholds for c@1
                best_lower_threshold: float | None = None
                best_upper_threshold: float | None = None
                best_n_answered_c_at_1 = -1
                best_value = -np.inf

                param_grid = ParameterGrid(
                    {"lower_threshold": threshold_values, "upper_threshold": threshold_values}
                )
                for params in param_grid:
                    lower_threshold = float(params["lower_threshold"])
                    upper_threshold = float(params["upper_threshold"])
                    if lower_threshold > upper_threshold:
                        continue
                    result = self.evaluate_at_threshold(
                        y_true=y_true_train,
                        scores=scores_train,
                        lower_threshold=lower_threshold,
                        upper_threshold=upper_threshold,
                        f1_threshold=None,
                    )
                    current_cat1_value = getattr(result, "c_at_1")

                    if current_cat1_value > best_value or (
                        np.isclose(current_cat1_value, best_value)
                        and result.n_answered_c_at_1 > best_n_answered_c_at_1
                    ):
                        best_value = current_cat1_value
                        best_n_answered_c_at_1 = result.n_answered_c_at_1
                        best_lower_threshold = lower_threshold
                        best_upper_threshold = upper_threshold

                logger.info(f"Finished split {len(per_split_results) + 1} of {n_splits} (optimizing across parameter "
                            f"grid).")
                assert best_lower_threshold is not None
                assert best_upper_threshold is not None
                # computes c@1 and AUROC for two optimizing thresholds, rest of metrics are computed on F1 threshold
                per_split_results.append(
                    self.evaluate_at_threshold(
                        y_true=y_true_test,
                        scores=scores_test,
                        lower_threshold=best_lower_threshold,
                        upper_threshold=best_upper_threshold,
                        f1_threshold=best_f1_threshold,
                    )
                )
        except ValueError as exc:
            raise ValueError(
                "Stratified split failed. Ensure each class in y_true has at least two samples."
            ) from exc

        metrics_mean, metrics_std, metric_values, metrics_ci = self._summarize_metrics(
            per_split_results, ci_level=ci_level, n_boot=n_boot
        )
        logger.info(f"Summary of metrics obtained. See metric means: {metrics_mean}")

        return EvaluationCVResult(
            split_config=split_config,
            candidate_thresholds=thresholds_list,
            per_split=per_split_results,
            metrics_mean=metrics_mean,
            metrics_std=metrics_std,
            metric_values=metric_values,
            metrics_ci=metrics_ci,
        )


def get_pan_metrics(predictions, y_true):
    """
    Compute PAN metrics for each method name and return a structured summary.
    Example values: predictions={"method_a": [0.1, 0.4]}, y_true=[0, 1].
    """
    evaluator = BinaryVerificationEvaluator()

    pan_metrics = {}

    for method_name, method_scores in predictions.items():
        logger.info(f"Starting with method {method_name}.")
        method_scores = np.asarray(method_scores, dtype=float)
        cv_result = evaluator.tune_thresholds(
                y_true=y_true,
                scores=method_scores,
                thresholds=CONFIG.THRESHOLDS,
                )

        pan_metrics[method_name] = {
                "split_config": cv_result.split_config,
                "candidate_thresholds": cv_result.candidate_thresholds,
                "metrics_mean": cv_result.metrics_mean,
                "metrics_std": cv_result.metrics_std,
                "metric_values": cv_result.metric_values,
                "per_split": [
                        # c@1 and AUROC (and auroc_c_at_1) are computed on the two c@1-optimized thresholds
                        # F1 threshold is used for all other metrics
                    {
                        "lower_threshold": result.lower_threshold,
                        "upper_threshold": result.upper_threshold,
                        "f1_threshold": result.f1_threshold,
                        "n_answered_c_at_1": result.n_answered_c_at_1,
                        "n_unanswered_c_at_1": result.n_unanswered_c_at_1,
                        "precision": result.precision,
                        "recall": result.recall,
                        "f1": result.f1,
                        "accuracy": result.accuracy,
                        "c_at_1": result.c_at_1,
                        "auroc": result.auroc,
                        "auroc_c_at_1": result.auroc_c_at_1,
                    }
                    for result in cv_result.per_split
                ],
                }
        logger.info(f"Obtained summary of PAN metrics for method {method_name}.")
    return pan_metrics

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

def _looks_like_pan_metrics(value: object) -> bool:
    return isinstance(value, dict) and (
        "metrics_mean" in value or "per_split" in value or "split_config" in value
    )

def _iter_pan_metrics_entries(pan_metrics: dict, dataset_name: str | None):
    if dataset_name is not None:
        for method_name, metrics in pan_metrics.items():
            if (
                isinstance(metrics, dict)
                and method_name in metrics
                and _looks_like_pan_metrics(metrics[method_name])
            ):
                yield dataset_name, method_name, metrics[method_name]
            else:
                yield dataset_name, method_name, metrics
        return

    known_datasets = {
        CONFIG.PAN20,
        CONFIG.PAN23,
        CONFIG.PAN25,
        CONFIG.BLOG,
        CONFIG.GUTENBERG,
        CONFIG.KOPPEL,
        CONFIG.STUDENT_ESSAYS,
        CONFIG.ARTIFICIAL_STUDENT_ESSAYS,
    }

    if all(key in known_datasets for key in pan_metrics.keys()):
        for dataset_key, dataset_metrics in pan_metrics.items():
            if not isinstance(dataset_metrics, dict):
                continue
            for method_name, metrics in dataset_metrics.items():
                if (
                    isinstance(metrics, dict)
                    and method_name in metrics
                    and _looks_like_pan_metrics(metrics[method_name])
                ):
                    yield dataset_key, method_name, metrics[method_name]
                else:
                    yield dataset_key, method_name, metrics
        return

    for method_name, metrics in pan_metrics.items():
        if _looks_like_pan_metrics(metrics):
            logger.warning(
                "Skipping MongoDB PAN metrics save for method '%s' because dataset_name was not provided.",
                method_name,
            )
        else:
            logger.warning(
                "Skipping MongoDB PAN metrics save for key '%s' due to unexpected structure.",
                method_name,
            )

def save_pan_metrics(pan_metrics, save_path, dataset_name=None, save_to_mongodb: bool = True):
    """
    Persist PAN metrics to JSON in the provided directory.
    Example values: pan_metrics={"method_a": {...}}, save_path=Path("results"),
    dataset_name="pan23" or None.
    """
    assert save_path.exists(), f"Directory {save_path} does not exist."
    serializable = _jsonify_value(pan_metrics)
    if dataset_name is None:
        save_file = save_path / "pan_metrics.json"
    else:
        save_file = save_path / f"{dataset_name}_pan_metrics.json"
    with open(save_file, "w") as f:
        json.dump(serializable, f, indent=2)
        logger.info(f"PAN metrics saved to {save_file}.")

    if save_to_mongodb:
        try:
            from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

            mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
            for dataset_key, method_name, metrics in _iter_pan_metrics_entries(
                pan_metrics=pan_metrics, dataset_name=dataset_name
            ):
                if dataset_key is None:
                    continue
                doc = {
                    "dataset_name": dataset_key,
                    "method_name": method_name,
                    "pan_metrics": _jsonify_value(metrics),
                }
                mongoDB.pan_metrics_collection.replace_one(
                    {"dataset_name": dataset_key, "method_name": method_name},
                    doc,
                    upsert=True,
                )
            logger.info("PAN metrics saved to MongoDB collection %s.", CONFIG.MONGO_PAN_METRICS_COLLECTION)
        except Exception as exc:
            logger.warning("Failed to save PAN metrics to MongoDB: %s", exc)
    return save_file
