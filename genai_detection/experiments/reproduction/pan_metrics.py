from __future__ import annotations

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


@dataclass
class EvaluationResult:
    threshold: float
    rejection_radius: float
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
    - score > threshold + rejection_radius   -> predict 1
    - score < threshold - rejection_radius   -> predict 0
    - otherwise                              -> unanswered (0.5)

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
    def predict_with_rejection(
        cls,
        scores: np.ndarray,
        threshold: float,
        rejection_radius: float = 0.0,
    ) -> np.ndarray:
        """
        Return predictions in {0, 1, 0.5}, where 0.5 means unanswered.
        """
        preds = np.full(shape=len(scores), fill_value=cls.UNANSWERED, dtype=float)
        preds[scores > threshold + rejection_radius] = 1.0
        preds[scores < threshold - rejection_radius] = 0.0
        return preds

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
        threshold: float,
        rejection_radius: float = 0.0,
    ) -> EvaluationResult:
        y_true_np = self._to_numpy(y_true).astype(int)
        scores_np = self._to_numpy(scores).astype(float)

        if len(y_true_np) != len(scores_np):
            raise ValueError("y_true and scores must have the same length.")

        if len(np.unique(y_true_np)) < 2:
            raise ValueError("AUROC requires both classes to be present in y_true.")

        y_pred_with_reject = self.predict_with_rejection(
            scores=scores_np,
            threshold=threshold,
            rejection_radius=rejection_radius,
        )

        y_true_answered, y_pred_answered = self._answered_only(
            y_true_np,
            y_pred_with_reject,
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

        c_at_1 = self.compute_c_at_1(y_true_np, y_pred_with_reject)
        auroc = roc_auc_score(y_true_np, scores_np)
        auroc_c_at_1 = auroc * c_at_1

        return EvaluationResult(
            threshold=threshold,
            rejection_radius=rejection_radius,
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
        rejection_radius: float = 0.0,
        optimize_for: str = "c_at_1",
    ) -> EvaluationResult:
        """
        Tune threshold on a validation set.

        Supported `optimize_for`:
        - "c_at_1"
        - "f1"
        - "accuracy"
        - "precision"
        - "recall"
        - "auroc_c_at_1"

        Returns the full evaluation result at the best threshold.
        """
        y_true_np = self._to_numpy(y_true).astype(int)
        scores_np = self._to_numpy(scores).astype(float)

        if len(y_true_np) != len(scores_np):
            raise ValueError("y_true and scores must have the same length.")

        if thresholds is None:
            unique_scores = np.unique(scores_np)
            thresholds = np.concatenate(
                (
                    [float(np.min(scores_np)) - 1e-12],
                    unique_scores,
                    [float(np.max(scores_np)) + 1e-12],
                )
            )

        best_result: EvaluationResult | None = None
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

        for threshold in thresholds:
            result = self.evaluate_at_threshold(
                y_true=y_true_np,
                scores=scores_np,
                threshold=float(threshold),
                rejection_radius=rejection_radius,
            )
            current_value = getattr(result, optimize_for)

            if current_value > best_value:
                best_value = current_value
                best_result = result

        assert best_result is not None
        return best_result