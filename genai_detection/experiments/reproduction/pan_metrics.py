from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_consistent_length

from genai_detection.config import CONFIG


@dataclass
class EvaluationResult:
    lower_threshold: float
    upper_threshold: float
    n_answered: int
    n_unanswered: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    c_at_1: float
    auroc: float
    auroc_c_at_1: float


class BinaryVerificationEvaluator:
    """
    Evaluate binary verification scores with threshold-dependent metrics,
    c@1, AUROC, and AUROC*c@1.

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


    @classmethod
    def compute_c_at_1(
        cls,
        y_true: np.ndarray,
        y_pred_with_reject: np.ndarray,
    ) -> float:
        """
        Compute c@1.

        `y_pred_with_reject` must contain:
        - 0 for negative prediction
        - 1 for positive prediction
        - 0.5 for unanswered
        """
        n = len(y_true)
        if n == 0:
            raise ValueError("Empty inputs are not allowed.")

        answered_mask = y_pred_with_reject != cls.UNANSWERED
        unanswered_mask = ~answered_mask

        n_u = int(np.sum(unanswered_mask))
        n_c = int(np.sum(y_true[answered_mask] == y_pred_with_reject[answered_mask]))

        return (1.0 / n) * (n_c + n_u * (n_c / n))

    @classmethod
    def _answered_only(
        cls,
        y_true: np.ndarray,
        y_pred_with_reject: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = y_pred_with_reject != cls.UNANSWERED
        return y_true[mask], y_pred_with_reject[mask].astype(int)

    def evaluate_at_threshold(
        self,
        y_true: Sequence[int] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        lower_threshold: float,
        upper_threshold: float,
    ) -> EvaluationResult:
        """
        Evaluates the performance of a binary classification model at a given a lower and upper threshold.
        Rejection means it is filled with a specific unanswered value.
        This method calculates various metrics such as precision,
        recall, F1-score, accuracy, c@1, AUROC, and a combined measure (AUROC*c@1).

        :param y_true: Ground truth binary labels. Must contain at least two unique classes.
        :type y_true: Sequence[int] | np.ndarray
        :param scores: Predicted scores or probabilities for the positive class. Must have
            the same length as `y_true`.
        :type scores: Sequence[float] | np.ndarray
        :param lower_threshold: Decision threshold below which predictions are categorized
            as the negative class.
        :param upper_threshold: Decision threshold above which predictions are categorized as positive class.
        :return: An `EvaluationResult` object containing thresholds, the number
            of answered and unanswered instances, and computed metrics like precision, recall,
            F1-score, accuracy, c@1, AUROC, and AUROC*c@1.
        :rtype: EvaluationResult
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

        n_answered = len(y_true_answered)
        n_unanswered = len(y_true_np) - n_answered

        if n_answered == 0:
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

        c_at_1 = self.compute_c_at_1(y_true_np, preds)
        auroc = roc_auc_score(y_true_np, scores_np)
        auroc_c_at_1 = auroc * c_at_1

        return EvaluationResult(
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            n_answered=n_answered,
            n_unanswered=n_unanswered,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            c_at_1=float(c_at_1),
            auroc=float(auroc),
            auroc_c_at_1=float(auroc_c_at_1),
        )

    def tune_threshold(
        self,
        y_true: Sequence[int] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        thresholds: Iterable[float] | None = None,
        optimize_for: str = "c_at_1",
    ) -> EvaluationResult:
        """
        Tune a threshold on a stratified subset of the input dataset and return
        results on the held-out items.
        A deterministic 50/50 split is used with class-stratification based on `y_true`.

        Two thresholds are selected directly from the candidate `thresholds` list.
        Pairs are evaluated with the constraint `lower_threshold <= upper_threshold`.
        Rejection means it is filled with a specific unanswered value.

        Supported `optimize_for`:
        - "c_at_1"
        - "f1"
        - "accuracy"
        - "precision"
        - "recall"
        - "auroc_c_at_1"

        Returns the full evaluation result at the best threshold.
        """
        y_true_np, scores_np = self._validate_binary_inputs(y_true=y_true, scores=scores)

        if len(np.unique(y_true_np)) < 2:
            raise ValueError("Threshold tuning requires both classes to be present in y_true.")

        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
            train_idx, test_idx = next(splitter.split(scores_np, y_true_np))
        except ValueError as exc:
            raise ValueError(
                "Stratified split failed. Ensure each class in y_true has at least two samples."
            ) from exc

        y_true_train = y_true_np[train_idx]
        scores_train = scores_np[train_idx]
        y_true_test = y_true_np[test_idx]
        scores_test = scores_np[test_idx]

        threshold_values = self._normalize_thresholds(scores=scores_train, thresholds=thresholds)

        best_lower_threshold: float | None = None
        best_upper_threshold: float | None = None
        best_n_answered = -1
        best_value = -np.inf

        valid_metrics = {
            "c_at_1",
            "f1",
            "accuracy",
            "precision",
            "recall",
            "auroc_c_at_1",
        }
        if optimize_for not in valid_metrics:
            raise ValueError(
                f"Unsupported optimize_for='{optimize_for}'. "
                f"Choose one of: {sorted(valid_metrics)}"
            )

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
            )
            current_value = getattr(result, optimize_for)

            if current_value > best_value or (
                np.isclose(current_value, best_value) and result.n_answered > best_n_answered
            ):
                best_value = current_value
                best_n_answered = result.n_answered
                best_lower_threshold = lower_threshold
                best_upper_threshold = upper_threshold

        assert best_lower_threshold is not None
        assert best_upper_threshold is not None
        return self.evaluate_at_threshold(
            y_true=y_true_test,
            scores=scores_test,
            lower_threshold=best_lower_threshold,
            upper_threshold=best_upper_threshold,
        )


def get_pan_metrics(predictions, y_true):
    evaluator = BinaryVerificationEvaluator()

    pan_metrics = {}

    for method_name, method_scores in predictions.items():
        method_scores = np.asarray(method_scores, dtype=float)

        print(Counter(method_scores))

        best_result = evaluator.tune_threshold(
                y_true=y_true,
                scores=method_scores,
                thresholds=CONFIG.THRESHOLDS,
                optimize_for="c_at_1",
                )

        pan_metrics[method_name] = {
                "upper_threshold": best_result.upper_threshold,
                "lower_threshold": best_result.lower_threshold,
                "n_answered": best_result.n_answered,
                "n_unanswered": best_result.n_unanswered,
                "precision": best_result.precision,
                "recall": best_result.recall,
                "f1": best_result.f1,
                "accuracy": best_result.accuracy,
                "c_at_1": best_result.c_at_1,
                "auroc": best_result.auroc,
                "auroc_c_at_1": best_result.auroc_c_at_1,
                }
    return pan_metrics

def save_pan_metrics(pan_metrics, save_path, dataset_name=None):
    assert save_path.exists(), f"Directory {save_path} does not exist."
    if dataset_name is None:
        save_file = save_path / "pan_metrics.json"
    else:
        save_file = save_path / f"{dataset_name}_pan_metrics.json"
    with open(save_file, "w") as f:
        import json
        json.dump(pan_metrics, f, indent=2)
    return save_file
