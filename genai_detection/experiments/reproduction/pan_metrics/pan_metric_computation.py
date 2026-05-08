from __future__ import annotations

"""Metric computation logic for PAN-style verification evaluation."""

from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import (
    accuracy_score,
    f1_score, precision_recall_curve, precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_consistent_length

import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Metrics for a single evaluation using a fixed (lower, upper) threshold pair.
    """

    lower_threshold: float = field(metadata={"summary": False})
    upper_threshold: float = field(metadata={"summary": False})
    f1_threshold: float | None = field(metadata={"summary": False})
    n_answered_c_at_1: int = field(metadata={"summary": False})
    n_unanswered_c_at_1: int = field(metadata={"summary": False})
    precision: float = field(metadata={"summary": True})
    recall: float = field(metadata={"summary": True})
    f1: float = field(metadata={"summary": True})
    accuracy: float = field(metadata={"summary": True})
    c_at_1: float = field(metadata={"summary": True})
    auroc: float = field(metadata={"summary": True})
    auroc_c_at_1: float = field(metadata={"summary": True})


@dataclass
class EvaluationCVResult:
    """
    Cross-validation summary for threshold tuning.

    - split_config: parameters used to generate CV splits.
    - per_split: list of EvaluationResult objects, one per fold.
    - metric_values: raw per-fold metric values (length = n_splits * n_repeats).
    - metrics_mean/std: summary statistics across folds.
    - metrics_ci: percentile bootstrap CI for the mean of each metric.
    """

    split_config: dict[str, float | int]
    per_split: list[EvaluationResult]
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    metric_values: dict[str, list[float]]
    metrics_ci: dict[str, dict[str, float | int | str]]


class PANMetricComputer:
    """
    Evaluate binary verification scores with threshold-dependent metrics.

    This class is data-agnostic: callers provide y_true and scores for any
    dataset subset. It computes per-fold evaluation metrics, supports tuning
    thresholds on repeated stratified k-fold splits, and returns dataclass
    results for downstream storage or significance testing.
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
        Example: y_true=[0, 1, 0], scores=[0.12, 0.87, 0.33].
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
    def _summary_metric_names() -> list[str]:
        return [
            f.name
            for f in fields(EvaluationResult)
            if f.metadata.get("summary", True)
        ]

    @staticmethod
    def extract_metric_values(
        results: Sequence[EvaluationResult] | Sequence[dict],
    ) -> dict[str, list[float]]:
        """
        Extract per-metric values across folds.
        Accepts EvaluationResult objects or dict-like results.
        """
        metric_values: dict[str, list[float]] = defaultdict(list)
        if not results:
            return {}

        first = results[0]
        if isinstance(first, EvaluationResult):
            for r in results:
                for f in fields(r):
                    if not f.metadata.get("summary", True):
                        continue
                    v = getattr(r, f.name)
                    metric_values[f.name].append(np.nan if v is None else float(v))
            return {metric: list(values) for metric, values in metric_values.items()}

        summary_names = PANMetricComputer._summary_metric_names()
        for r in results:
            for name in summary_names:
                v = r.get(name)
                metric_values[name].append(np.nan if v is None else float(v))
        return {metric: list(values) for metric, values in metric_values.items()}

    @staticmethod
    def _extract_metric_values(
        results: Sequence[EvaluationResult] | Sequence[dict],
    ) -> dict[str, list[float]]:
        """Backward-compatible alias for legacy code paths."""
        return PANMetricComputer.extract_metric_values(results)

    @staticmethod
    def _summarize_metrics(
        results: Sequence[EvaluationResult],
        ci_level: float,
        n_boot: int,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, list[float]],
        dict[str, dict[str, float | int | str]],
    ]:
        """
        Compute per-metric mean/std across folds and percentile bootstrap CIs
        for the mean.
        """
        metric_values = PANMetricComputer.extract_metric_values(results)

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
            metrics_ci[metric] = {
                "low": float(boot.confidence_interval.low),
                "high": float(boot.confidence_interval.high),
                "level": float(ci_level),
                "method": "bootstrap_percentile",
                "n_boot": int(n_boot),
            }

        metric_values_dict = {metric: list(values) for metric, values in metric_values.items()}

        return metrics_mean, metrics_std, metric_values_dict, metrics_ci

    @classmethod
    def compute_c_at_1(
        cls,
        y_true: np.ndarray,
        y_preds: np.ndarray,
    ) -> float:
        """
        Compute c@1 as defined in PAN-style evaluation.
        Peñas & Rodrigo (2011): "A Simple Measure to Assess Non-response".
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
    ) -> tuple[float, float]:

        precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)

        prec = precision[1:]
        rec = recall[1:]
        thr = pr_thresholds

        # avoid division by zero
        f1 = 2 * (prec * rec) / (prec + rec + 1e-12)

        # apply your constraint: exclude degenerate cases
        mask = (prec > 0) & (prec < 1) & (rec > 0) & (rec < 1)
        if not np.any(mask):
            best = int(np.argmax(f1))  # fallback to unconstrained best
            return float(thr[best]), float(f1[best])

        best = int(np.argmax(f1[mask]))
        return float(thr[mask][best]), float(f1[mask][best])

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
            if n_answered_c_at_1 == 0:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                accuracy = 0.0
            else:
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
                f1 = f1_score(
                    y_true_answered,
                    y_pred_answered,
                    zero_division=0,
                )
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
        n_splits: int = 10,
        n_repeats: int = 5,
        random_state: int = 42,
        ci_level: float = 0.95,
        n_boot: int = 10000,
        splits: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> EvaluationCVResult:
        """
        Tune thresholds on repeated stratified k-fold splits and summarize
        performance on held-out folds.
        """
        y_true_np, scores_np = self._validate_binary_inputs(y_true=y_true, scores=scores)

        class_counts = np.bincount(y_true_np)
        if (class_counts < n_splits).any():
            raise ValueError(
                f"Each class must have at least n_splits={n_splits} samples. "
                f"Counts: {class_counts.tolist()}"
            )
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

        if splits is None:
            # cannot access split manager here, so we need to do it manually
            splitter = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )
            split_iter = splitter.split(scores_np, y_true_np)
            logger.info("Splitting data with %s.", split_config)
        else:
            splits = list(splits)
            expected = int(n_splits) * int(n_repeats)
            if len(splits) != expected:
                raise ValueError(
                    f"Expected {expected} splits (n_splits={n_splits}, n_repeats={n_repeats}), "
                    f"got {len(splits)}."
                )
            split_iter = splits
            logger.info("Using precomputed splits with %s.", split_config)

        for split_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
            y_true_train = y_true_np[train_idx]
            scores_train = scores_np[train_idx]
            y_true_test = y_true_np[test_idx]
            scores_test = scores_np[test_idx]

            threshold_values = self._normalize_thresholds(scores=scores_train, thresholds=None)
            max_thresholds = 300
            if threshold_values.size > max_thresholds:
                logger.warning(
                    "Too many thresholds (%d) for optimization. Using %d thresholds instead.",
                    threshold_values.size,
                    max_thresholds,
                )
                # Create evenly spaced quantile levels between 1% and 99%,
                # then compute the corresponding score thresholds from the training scores.
                # np.unique removes duplicate threshold values.
                qs = np.linspace(0.01, 0.99, max_thresholds)
                threshold_values = np.unique(np.quantile(scores_train, qs))

            best_f1_threshold, _ = self._optimize_f1_threshold(
                y_true=y_true_train,
                scores=scores_train
            )

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
                # give F1 none value to avoid computing F1, recall, precision each time
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

            logger.info("Finished split %d/%d", split_idx, n_splits * n_repeats)
            assert best_lower_threshold is not None, f"Best lower c@1 threshold is None for split {split_idx}."
            assert best_upper_threshold is not None, f"Best upper c@1 threshold is None for split {split_idx}."

            # obtain final scores using tunes F1 (for F1, recall, precision) and c@1 (for c@1) thresholds
            per_split_results.append(
                self.evaluate_at_threshold(
                    y_true=y_true_test,
                    scores=scores_test,
                    lower_threshold=best_lower_threshold,
                    upper_threshold=best_upper_threshold,
                    f1_threshold=best_f1_threshold,
                )
            )

        metrics_mean, metrics_std, metric_values, metrics_ci = self._summarize_metrics(
            per_split_results, ci_level=ci_level, n_boot=n_boot
        )
        return EvaluationCVResult(
            split_config=split_config,
            per_split=per_split_results,
            metrics_mean=metrics_mean,
            metrics_std=metrics_std,
            metric_values=metric_values,
            metrics_ci=metrics_ci,
        )


# Backwards-compatible alias for existing code that expects BinaryVerificationEvaluator.
BinaryVerificationEvaluator = PANMetricComputer

__all__ = [
    "EvaluationResult",
    "EvaluationCVResult",
    "PANMetricComputer",
    "BinaryVerificationEvaluator",
]
