from __future__ import annotations

"""Cross-validation helpers for PAN metrics."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import EvaluationCVResult, PANMetricComputer


@dataclass
class SplitManager:
    """
    Deterministic repeated stratified k-fold split generation.

    Store split configuration so it can be reused across methods for paired
    significance tests.
    """

    n_splits: int = 10
    n_repeats: int = 5
    random_state: int = 42

    def build_splits(self, y_true: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        splitter = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        y_true = np.asarray(y_true)
        return list(splitter.split(np.zeros_like(y_true), y_true))

    def split_config(self) -> dict[str, float | int]:
        return {
            "n_splits": int(self.n_splits),
            "n_repeats": int(self.n_repeats),
            "test_size": float(1.0 / self.n_splits),
            "random_state": int(self.random_state),
        }


class PANEvaluator:
    """
    Orchestrates PAN metric computation with consistent splits.
    """

    def __init__(self, metric_computer: PANMetricComputer, split_manager: SplitManager) -> None:
        self.metric_computer = metric_computer
        self.split_manager = split_manager

    def compute_cv(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        *,
        ci_level: float = 0.95,
        n_boot: int = 10000,
        splits: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> EvaluationCVResult:
        if splits is None:
            splits = self.split_manager.build_splits(y_true)
        return self.metric_computer.tune_thresholds(
            y_true=y_true,
            scores=scores,
            n_splits=self.split_manager.n_splits,
            n_repeats=self.split_manager.n_repeats,
            random_state=self.split_manager.random_state,
            ci_level=ci_level,
            n_boot=n_boot,
            splits=splits,
        )

    def extract_metric_values(
        self,
        per_split: list[dict] | list,
    ) -> dict[str, list[float]]:
        return self.metric_computer.extract_metric_values(per_split)
