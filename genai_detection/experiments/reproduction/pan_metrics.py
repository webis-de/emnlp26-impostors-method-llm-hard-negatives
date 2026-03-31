from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from itertools import combinations
import json
import os
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import bootstrap, mannwhitneyu, ttest_ind, ttest_rel, wilcoxon
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
        # stores metric values as lists for Bootstrapping code
        metric_values = defaultdict(list)
        for r in results:
            for f in fields(r):
                # thresholds should not part of summary (i.e., computing mean and std makes no sense)
                if not f.metadata.get("summary", True):
                    continue
                v = getattr(r, f.name)
                metric_values[f.name].append(np.nan if v is None else float(v))

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

def save_pan_metrics_to_mongo_db(pan_metrics, dataset_name:str):
    try:
        from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

        mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
        for method_name, metrics in _iter_pan_metrics_entries(
            pan_metrics=pan_metrics
        ):
            doc = {
                "dataset_name": dataset_name,
                "method_name": method_name,
                "pan_metrics": _jsonify_value(metrics),
            }
            mongoDB.pan_metrics_collection.replace_one(
                {"dataset_name": dataset_name, "method_name": method_name},
                doc,
                upsert=True,
            )
        logger.info(
            "PAN metrics saved to MongoDB collection %s.",
            CONFIG.MONGO_PAN_METRICS_COLLECTION,
        )
    except Exception as exc:
        logger.warning("Failed to save PAN metrics to MongoDB: %s", exc)


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
        pan_metrics[method_name] = asdict(cv_result)

        # pan_metrics[method_name] = {
        #         "split_config": cv_result.split_config,
        #         "candidate_thresholds": cv_result.candidate_thresholds,
        #         "metrics_mean": cv_result.metrics_mean,
        #         "metrics_std": cv_result.metrics_std,
        #         "metric_values": cv_result.metric_values,
        #         "per_split": [
        #                 # c@1 and AUROC (and auroc_c_at_1) are computed on the two c@1-optimized thresholds
        #                 # F1 threshold is used for all other metrics
        #             {
        #                 "lower_threshold": result.lower_threshold,
        #                 "upper_threshold": result.upper_threshold,
        #                 "f1_threshold": result.f1_threshold,
        #                 "n_answered_c_at_1": result.n_answered_c_at_1,
        #                 "n_unanswered_c_at_1": result.n_unanswered_c_at_1,
        #                 "precision": result.precision,
        #                 "recall": result.recall,
        #                 "f1": result.f1,
        #                 "accuracy": result.accuracy,
        #                 "c_at_1": result.c_at_1,
        #                 "auroc": result.auroc,
        #                 "auroc_c_at_1": result.auroc_c_at_1,
        #             }
        #             for result in cv_result.per_split
        #         ],
        #         }
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

def _iter_pan_metrics_entries(pan_metrics: dict):
    for method_name, metrics in pan_metrics.items():
        if (
            isinstance(metrics, dict)
            and method_name in metrics
            and _looks_like_pan_metrics(metrics[method_name])
        ):
            yield method_name, metrics[method_name]
        else:
            yield method_name, metrics
    return

def save_pan_metrics(pan_metrics, save_path, dataset_name=None):
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
    return save_file

def _filter_finite_pairs(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _pairwise_p_value(
    a: np.ndarray,
    b: np.ndarray,
    test: str,
    alternative: str,
) -> float:
    if test == "wilcoxon":
        stat = wilcoxon(a, b, alternative=alternative, zero_method="pratt")
        return float(stat.pvalue)
    if test == "ttest_rel":
        stat = ttest_rel(a, b, alternative=alternative)
        return float(stat.pvalue)
    if test == "mannwhitney":
        stat = mannwhitneyu(a, b, alternative=alternative)
        return float(stat.pvalue)
    if test == "ttest_ind":
        stat = ttest_ind(a, b, alternative=alternative)
        return float(stat.pvalue)
    raise ValueError(f"Unknown test '{test}'.")


def compare_pan_metrics_significance(
    pan_metrics: dict,
    metrics: Iterable[str] | None = None,
    alpha_levels: Sequence[float] = (0.05, 0.01, 0.005),
    test: str = "wilcoxon",
    alternative: str = "two-sided",
    min_samples: int = 2,
) -> dict[str, dict]:
    """
    Assess pairwise significance between approaches per metric using per-fold values.

    - pan_metrics: mapping of method -> metrics, i.e., no dataset_name as key (should only contain values for one dataset).
    - dataset_name: select a dataset when pan_metrics is dataset-keyed.
    - metrics: subset of metric names; defaults to metrics present in metric_values.
    - alpha_levels: p-value thresholds to report.
    - test: one of {"wilcoxon", "ttest_rel", "mannwhitney", "ttest_ind"}.
    - alternative: scipy.stats alternative hypothesis parameter.
    - min_samples: minimum paired samples required to run a test.

    Returns dict with per-pair, per-metric p-values and significance flags.
    """
    method_names = sorted(pan_metrics.keys())
    if len(method_names) < 2:
        raise ValueError("Need at least two methods to compare.")

    alpha_sorted = sorted(set(float(a) for a in alpha_levels))
    results: dict[str, dict] = {
        "test": test,
        "alternative": alternative,
        "alpha_levels": alpha_sorted,
        "pairs": {},
    }

    for method_a, method_b in combinations(method_names, 2):
        metrics_a = pan_metrics[method_a].get("metric_values", {})
        metrics_b = pan_metrics[method_b].get("metric_values", {})
        if not metrics_a or not metrics_b:
            raise ValueError(
                f"Missing metric_values for methods '{method_a}' or '{method_b}'."
            )

        if metrics is None:
            metric_names = sorted(set(metrics_a.keys()) & set(metrics_b.keys()))
        else:
            metric_names = list(metrics)

        pair_key = f"{method_a} vs {method_b}"
        pair_results: dict[str, dict] = {}

        for metric in metric_names:
            values_a = np.asarray(metrics_a.get(metric, []), dtype=float)
            values_b = np.asarray(metrics_b.get(metric, []), dtype=float)

            if test in {"mannwhitney", "ttest_ind"}:
                values_a = values_a[np.isfinite(values_a)]
                values_b = values_b[np.isfinite(values_b)]
            else:
                values_a, values_b = _filter_finite_pairs(values_a, values_b)
            if len(values_a) != len(values_b) and test in {"wilcoxon", "ttest_rel"}:
                raise ValueError(
                    f"Paired test '{test}' requires equal-length samples for "
                    f"metric '{metric}' ({method_a} vs {method_b})."
                )
            if len(values_a) < min_samples or len(values_b) < min_samples:
                pair_results[metric] = {
                    "p_value": float("nan"),
                    "n": int(min(len(values_a), len(values_b))),
                    "significant": {str(a): False for a in alpha_sorted},
                }
                continue

            try:
                p_value = _pairwise_p_value(values_a, values_b, test, alternative)
            except ValueError:
                p_value = float("nan")

            pair_results[metric] = {
                "p_value": float(p_value),
                "n": int(min(len(values_a), len(values_b))),
                "significant": {str(a): bool(p_value <= a) for a in alpha_sorted},
            }

        results["pairs"][pair_key] = pair_results

    return results
