# Copyright 2026 Klara M. Gutekunst, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate statistical-inference impostor predictions with PAN metrics."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from bson import ObjectId
from sklearn.metrics import cohen_kappa_score

from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.experiments.reproduction.pan_metrics.pan_cv import (
    SplitManager,
    compute_aligned_pair_keys_hash,
)
from genai_detection.experiments.reproduction.pan_metrics.pan_data_loader import (
    PANDataLoader,
)
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import (
    EvaluationCVResult,
    PANMetricComputer,
)
from genai_detection.experiments.reproduction.pan_metrics.run_pan_metrics import (
    CI_LEVEL,
    N_BOOT,
    N_IMPOSTORS,
    N_POTENTIAL_IMPOSTORS,
    N_REPEATS,
    N_SPLITS,
    RANDOM_STATE,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

DEFAULT_PREDICTION_FIELD = "pred_both_hypotheses_directions"
RIGHT_DISPUTED_LEFT_CANDIDATE_PREDICTION_FIELD = (
    "right_disputed_left_candidate_uncorrected_p_value_pred"
)
LEFT_DISPUTED_RIGHT_CANDIDATE_PREDICTION_FIELD = (
    "left_disputed_right_candidate_uncorrected_p_value_pred"
)
RIGHT_DISPUTED_LEFT_CANDIDATE_CORRECTED_P_VALUE_FIELD = (
    "right_disputed_left_candidate_corrected_p_value"
)
LEFT_DISPUTED_RIGHT_CANDIDATE_CORRECTED_P_VALUE_FIELD = (
    "left_disputed_right_candidate_corrected_p_value"
)
KAPPA_LATEX_OUTPUT_FILENAME = (
    "kappa_annotator_agreement_directional_uncorrected_p_value_predictions.tex"
)
ALPHA_SWEEP_DIRNAME = "alpha_sweeps"
DEFAULT_ALPHA_MIN = 0.001
DEFAULT_ALPHA_MAX = 0.9
DEFAULT_N_ALPHA = 30
METRIC_ORDER = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auroc",
    "c_at_1",
    "auroc_c_at_1",
]

METRIC_LABELS = {
    "accuracy": "Acc.",
    "f1": "F1",
    "precision": "Prec.",
    "recall": "Rec.",
    "auroc": "AUROC",
    "c_at_1": "c@1",
    "auroc_c_at_1": (
        r"\raisebox{0.75ex}[0em][0em]{\begin{tabular}{@{}c@{}}AUROC\\[-0.5ex]$\times$~c@1\end{tabular}}"
    ),
}

STATISTICAL_INFERENCE_DATASET_LABELS = {
    CONFIG.BLOG: "Blogs Posts Dataset",
    CONFIG.STUDENT_ESSAYS: "Student Essays Dataset",
}

# TODO: delete
# STATISTICAL_INFERENCE_METHOD_LABELS = {
#     "on_the_fly_chatnoir": "ChatNoir",
#     "on_the_fly_startpage": "Startpage",
#     "in_domain": "In-Domain",
#     "two_step_llm": "LLM impostors",
#     "one_step_llm",
#     "random_words",
# }

DEFAULT_METHOD_ORDER = [
    "on_the_fly_chatnoir",
    "on_the_fly_startpage",
    "in_domain",
    "two_step_llm",
    "one_step_llm",
    "random_words",
    # "ppmd",
    # "unmasking",
    # "supervised_baseline",
    # "unsupervised_baseline_cosine",
    # "unsupervised_baseline_min-max",
    # "bdi",
    # "homotopy",
    # "potha2017",
    # "asgalf",
    # "std_impostor",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_dir() -> Path:
    return _repo_root() / CONFIG.SAVE_PATH / "statistical_inference"


def _safe_filename(value: str) -> str:
    safe_chars = [c if c.isalnum() or c in {"-", "_"} else "_" for c in value]
    return "".join(safe_chars).strip("_") or "predictions"


def _alpha_grid(alpha_min: float, alpha_max: float, num_points: int = 30) -> list[float]:
    if alpha_min <= 0:
        raise ValueError("--alpha-min must be greater than 0.")
    if alpha_max <= alpha_min:
        raise ValueError("--alpha-max must be greater than --alpha-min.")
    if num_points < 2:
        raise ValueError("--num-points must be at least 2.")

    return np.logspace(
        np.log10(alpha_min),
        np.log10(alpha_max),
        num_points,
    ).tolist()


def _format_alpha(alpha: float) -> str:
    return f"{alpha:.3f}"


def _alpha_slug(alpha: float) -> str:
    return _safe_filename(f"alpha_{_format_alpha(alpha)}")


def _with_alpha_suffix(
    path: Path, alpha: float, *, directory: Path | None = None
) -> Path:
    parent = directory or path.parent
    return parent / f"{path.stem}_{_alpha_slug(alpha)}{path.suffix}"


def _with_shared_subset_suffix(path: Path, require_all_methods: bool) -> Path:
    return path.with_name(
        f"{path.stem}_shared_subset_{require_all_methods}{path.suffix}"
    )


def _prediction_field_for_alpha(alpha: float) -> str:
    return f"{DEFAULT_PREDICTION_FIELD}_{_alpha_slug(alpha)}"


def _label_with_alpha(label: str, alpha: float) -> str:
    return f"{label}-{_alpha_slug(alpha).replace('_', '-')}"


def _caption_with_alpha(caption_template: str, alpha: float) -> str:
    alpha_value = _format_alpha(alpha)
    if "{alpha}" in caption_template:
        return caption_template.replace("{alpha}", alpha_value)
    return f"{caption_template} " + rf"($\alpha={alpha_value}$)."


def _display_dataset(dataset_name: str) -> str:
    return CONFIG.DATASET_TRANSLATIONS.get(dataset_name, dataset_name)


def _display_method(method_name: str) -> str:
    return CONFIG.LABEL_TRANSLATIONS.get(method_name, method_name)


def _display_statistical_inference_dataset(dataset_name: str) -> str:
    label = STATISTICAL_INFERENCE_DATASET_LABELS.get(
        dataset_name,
        _display_dataset(dataset_name),
    )
    if not label.lower().endswith("dataset"):
        label = f"{label} Dataset"
    return label


def _display_statistical_inference_method(method_name: str) -> str:
    return CONFIG.LABEL_TRANSLATIONS.get(
        method_name, _display_method(method_name)
    )


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _classification_error_rates(
    y_true: Sequence[int] | np.ndarray,
    predictions: Sequence[float] | np.ndarray,
) -> dict[str, float | int | None]:
    y_true_np = np.asarray(y_true).astype(int)
    predictions_np = np.asarray(predictions).astype(int)
    different_author_mask = y_true_np == 0
    same_author_mask = y_true_np == 1
    n_different_author = int(np.sum(different_author_mask))
    n_same_author = int(np.sum(same_author_mask))

    type_i_error = None
    if n_different_author:
        type_i_error = float(np.mean(predictions_np[different_author_mask] == 1))

    type_ii_error = None
    if n_same_author:
        type_ii_error = float(np.mean(predictions_np[same_author_mask] == 0))

    return {
        "type_i_error": type_i_error,
        "type_ii_error": type_ii_error,
        "n_different_author": n_different_author,
        "n_same_author": n_same_author,
    }


def _format_metric(value: Any, *, bold: bool = False) -> str:
    number = _float_or_none(value)
    if number is None:
        return "--"
    formatted = f"{number:.3f}"
    if bold:
        return r"\textbf{" + formatted + "}"
    return formatted


@dataclass(frozen=True)
class StatisticalInferenceMetrics:
    dataset_name: str
    method_name: str
    prediction_field: str
    alpha: float | None
    left_p_value_field: str | None
    right_p_value_field: str | None
    type_i_error: float | None
    type_ii_error: float | None
    n_different_author: int | None
    n_same_author: int | None
    n_samples: int
    aligned_pair_keys_hash: str
    split_config: dict[str, float | int]
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    metrics_ci: dict[str, dict[str, float | int | str]]
    metric_values: dict[str, list[float]]


@dataclass(frozen=True)
class StatisticalInferenceAgreement:
    dataset_name: str
    method_name: str
    right_prediction_field: str
    left_prediction_field: str
    alpha: float | None
    right_p_value_field: str | None
    left_p_value_field: str | None
    n_samples: int
    aligned_pair_keys_hash: str
    cohen_kappa: float | None
    observed_agreement: float | None


class StatisticalInferencePredictionLoader(PANDataLoader):
    """
    Load hard statistical-inference predictions from impostors_outputs.

    This mirrors PANDataLoader._load_impostor_scores, but projects a binary
    prediction field such as corr_pred_over_different_rounds instead of the
    continuous scores_over_different_rounds field.
    """

    @staticmethod
    def _build_impostor_output_query(
        technique: str,
        dataset_name: str,
        *,
        n_impostors: int,
        n_potential_impostors: int | None = None,
    ) -> dict[str, Any]:
        if technique not in IMPOSTOR_GENERATORS and "on_the_fly" not in technique:
            raise ValueError(f"Unsupported impostor technique: {technique}")

        if "on_the_fly" in technique and technique != "on_the_fly":
            index = CONFIG.RETRIEVAL_INDEX_TRANSLATIONS.get(technique)
            if index is None:
                raise ValueError(f"Unknown retrieval index for {technique}.")
            technique = "on_the_fly"
        else:
            index = None

        query: dict[str, Any] = {
            "impostor_generation_technique": technique,
            "n_impostors": n_impostors,
            "dataset_name": dataset_name,
        }
        if index is not None:
            query["retrieval_index"] = index
        if n_potential_impostors is not None:
            query["n_potential_impostors"] = n_potential_impostors
        return query

    def load_predictions(
        self,
        method_name: str,
        dataset_name: str,
        *,
        prediction_field: str = DEFAULT_PREDICTION_FIELD,
        n_impostors: int = N_IMPOSTORS,
        n_potential_impostors: int | None = N_POTENTIAL_IMPOSTORS,
    ) -> dict[tuple[ObjectId, ObjectId], float]:
        return self._load_impostor_predictions(
            technique=method_name,
            dataset_name=dataset_name,
            prediction_field=prediction_field,
            n_impostors=n_impostors,
            n_potential_impostors=n_potential_impostors,
        )

    def load_predictions_for_methods(
        self,
        methods: Iterable[str],
        dataset_name: str,
        *,
        prediction_field: str = DEFAULT_PREDICTION_FIELD,
        n_impostors: int = N_IMPOSTORS,
        n_potential_impostors: int | None = N_POTENTIAL_IMPOSTORS,
    ) -> dict[str, dict[tuple[ObjectId, ObjectId], float]]:
        return {
            method_name: self.load_predictions(
                method_name,
                dataset_name,
                prediction_field=prediction_field,
                n_impostors=n_impostors,
                n_potential_impostors=n_potential_impostors,
            )
            for method_name in methods
        }

    def _load_impostor_predictions(
        self,
        technique: str,
        dataset_name: str,
        *,
        prediction_field: str,
        n_impostors: int,
        n_potential_impostors: int | None = None,
    ) -> dict[tuple[ObjectId, ObjectId], float]:
        query = self._build_impostor_output_query(
            technique,
            dataset_name,
            n_impostors=n_impostors,
            n_potential_impostors=n_potential_impostors,
        )

        cursor = self.mongo.impostor_output_collection.find(
            query,
            {"left_id": 1, "right_id": 1, prediction_field: 1},
            batch_size=self.batch_size,
        ).sort("_id", 1)

        predictions_by_pair: dict[tuple[ObjectId, ObjectId], float] = {}
        missing_prediction_field = 0
        for doc in cursor:
            pair = (doc["left_id"], doc["right_id"])
            if pair in predictions_by_pair:
                continue
            if prediction_field not in doc:
                missing_prediction_field += 1
                continue
            predictions_by_pair[pair] = self._coerce_binary_prediction(
                doc[prediction_field],
                prediction_field=prediction_field,
            )

        if missing_prediction_field:
            logger.warning(
                "Skipped %d %s/%s documents without prediction field %s.",
                missing_prediction_field,
                dataset_name,
                technique,
                prediction_field,
            )

        return predictions_by_pair

    def load_directional_p_values(
        self,
        method_name: str,
        dataset_name: str,
        *,
        left_p_value_field: str = LEFT_DISPUTED_RIGHT_CANDIDATE_CORRECTED_P_VALUE_FIELD,
        right_p_value_field: str = RIGHT_DISPUTED_LEFT_CANDIDATE_CORRECTED_P_VALUE_FIELD,
        n_impostors: int = N_IMPOSTORS,
        n_potential_impostors: int | None = N_POTENTIAL_IMPOSTORS,
    ) -> dict[tuple[ObjectId, ObjectId], tuple[float, float]]:
        return self._load_impostor_p_values(
            technique=method_name,
            dataset_name=dataset_name,
            left_p_value_field=left_p_value_field,
            right_p_value_field=right_p_value_field,
            n_impostors=n_impostors,
            n_potential_impostors=n_potential_impostors,
        )

    def _load_impostor_p_values(
        self,
        technique: str,
        dataset_name: str,
        *,
        left_p_value_field: str,
        right_p_value_field: str,
        n_impostors: int,
        n_potential_impostors: int | None = None,
    ) -> dict[tuple[ObjectId, ObjectId], tuple[float, float]]:
        query = self._build_impostor_output_query(
            technique,
            dataset_name,
            n_impostors=n_impostors,
            n_potential_impostors=n_potential_impostors,
        )

        cursor = self.mongo.impostor_output_collection.find(
            query,
            {
                "left_id": 1,
                "right_id": 1,
                left_p_value_field: 1,
                right_p_value_field: 1,
            },
            batch_size=self.batch_size,
        ).sort("_id", 1)

        p_values_by_pair: dict[tuple[ObjectId, ObjectId], tuple[float, float]] = {}
        missing_p_value_fields = 0
        invalid_p_values = 0
        for doc in cursor:
            pair = (doc["left_id"], doc["right_id"])
            if pair in p_values_by_pair:
                continue
            if left_p_value_field not in doc or right_p_value_field not in doc:
                missing_p_value_fields += 1
                continue
            left_p_value = _float_or_none(doc.get(left_p_value_field))
            right_p_value = _float_or_none(doc.get(right_p_value_field))
            if left_p_value is None or right_p_value is None:
                invalid_p_values += 1
                continue
            p_values_by_pair[pair] = (left_p_value, right_p_value)

        if missing_p_value_fields:
            logger.warning(
                "Skipped %d %s/%s documents without p-value fields %s/%s.",
                missing_p_value_fields,
                dataset_name,
                technique,
                left_p_value_field,
                right_p_value_field,
            )
        if invalid_p_values:
            logger.warning(
                "Skipped %d %s/%s documents with invalid p-values in %s/%s.",
                invalid_p_values,
                dataset_name,
                technique,
                left_p_value_field,
                right_p_value_field,
            )

        return p_values_by_pair

    def load_directional_p_values_for_methods(
        self,
        methods: Iterable[str],
        dataset_name: str,
        *,
        left_p_value_field: str = LEFT_DISPUTED_RIGHT_CANDIDATE_CORRECTED_P_VALUE_FIELD,
        right_p_value_field: str = RIGHT_DISPUTED_LEFT_CANDIDATE_CORRECTED_P_VALUE_FIELD,
        n_impostors: int = N_IMPOSTORS,
        n_potential_impostors: int | None = N_POTENTIAL_IMPOSTORS,
    ) -> dict[str, dict[tuple[ObjectId, ObjectId], tuple[float, float]]]:
        return {
            method_name: self.load_directional_p_values(
                method_name,
                dataset_name,
                left_p_value_field=left_p_value_field,
                right_p_value_field=right_p_value_field,
                n_impostors=n_impostors,
                n_potential_impostors=n_potential_impostors,
            )
            for method_name in methods
        }

    @staticmethod
    def _coerce_binary_prediction(value: Any, *, prediction_field: str) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            value_f = float(value)
            if value_f in {0.0, 1.0}:
                return value_f
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return 1.0
            if normalized in {"false", "0"}:
                return 0.0
        raise ValueError(
            f"Prediction field {prediction_field!r} must contain binary values, got {value!r}."
        )


class FixedPredictionPANEvaluator:
    """
    Compute PAN metrics for already-thresholded 0/1 predictions.

    For c@1, a fixed lower/upper threshold of 0.5 means all predictions are
    answered. For F1, precision, recall, and accuracy, the same 0.5 threshold
    is used instead of tuning over a hard prediction field.
    """

    def __init__(
        self, metric_computer: PANMetricComputer, split_manager: SplitManager
    ) -> None:
        self.metric_computer = metric_computer
        self.split_manager = split_manager

    def compute_cv(
        self,
        y_true: Sequence[int] | np.ndarray,
        predictions: Sequence[float] | np.ndarray,
        *,
        ci_level: float = CI_LEVEL,
        n_boot: int = N_BOOT,
        splits: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> EvaluationCVResult:
        if self.split_manager.n_splits < 2:
            raise ValueError("n_splits must be >= 2 for cross-validation.")
        if self.split_manager.n_repeats < 1:
            raise ValueError("n_repeats must be >= 1 for cross-validation.")
        if not 0.0 < ci_level < 1.0:
            raise ValueError("ci_level must be between 0 and 1.")
        if n_boot < 1:
            raise ValueError("n_boot must be >= 1.")

        y_true_np = np.asarray(y_true).astype(int)
        predictions_np = np.asarray(predictions).astype(float)
        if y_true_np.shape[0] != predictions_np.shape[0]:
            raise ValueError(
                f"y_true and predictions must have same length, got {len(y_true_np)} and {len(predictions_np)}."
            )
        if not np.isin(predictions_np, [0.0, 1.0]).all():
            raise ValueError("predictions must be binary 0/1 values.")

        class_counts = np.bincount(y_true_np, minlength=2)
        if (class_counts < self.split_manager.n_splits).any():
            raise ValueError(
                f"Each class must have at least n_splits={self.split_manager.n_splits} samples. "
                f"Counts: {class_counts.tolist()}"
            )

        splits = (
            list(splits)
            if splits is not None
            else self.split_manager.build_splits(y_true_np)
        )
        expected_splits = self.split_manager.n_splits * self.split_manager.n_repeats
        if len(splits) != expected_splits:
            raise ValueError(f"Expected {expected_splits} splits, got {len(splits)}.")
        per_split_results = [
            self.metric_computer.evaluate_at_threshold(
                y_true=y_true_np[test_idx],
                scores=predictions_np[test_idx],
                lower_threshold=0.5,
                upper_threshold=0.5,
                f1_threshold=0.5,
            )
            for _, test_idx in splits
        ]

        metrics_mean, metrics_std, metric_values, metrics_ci = (
            PANMetricComputer._summarize_metrics(
                per_split_results,
                ci_level=ci_level,
                n_boot=n_boot,
            )
        )
        return EvaluationCVResult(
            split_config=self.split_manager.split_config(),
            per_split=per_split_results,
            metrics_mean=metrics_mean,
            metrics_std=metrics_std,
            metric_values=metric_values,
            metrics_ci=metrics_ci,
        )


class StatisticalInferenceExperiment:
    def __init__(
        self,
        loader: StatisticalInferencePredictionLoader,
        evaluator: FixedPredictionPANEvaluator,
    ) -> None:
        self.loader = loader
        self.evaluator = evaluator

    def evaluate_method(
        self,
        method_name: str,
        *,
        dataset_name: str,
        prediction_field: str,
        n_impostors: int,
        n_potential_impostors: int | None,
        ci_level: float,
        n_boot: int,
        predictions_by_pair: dict[tuple[ObjectId, ObjectId], float] | None = None,
        required_pair_keys: set[tuple[ObjectId, ObjectId]] | None = None,
    ) -> StatisticalInferenceMetrics | None:
        if predictions_by_pair is None:
            predictions_by_pair = self.loader.load_predictions(
                method_name,
                dataset_name,
                prediction_field=prediction_field,
                n_impostors=n_impostors,
                n_potential_impostors=n_potential_impostors,
            )
        if not predictions_by_pair:
            logger.warning(
                "No predictions found for method=%s dataset=%s field=%s.",
                method_name,
                dataset_name,
                prediction_field,
            )
            return None

        pair_keys = sorted(
            required_pair_keys
            if required_pair_keys is not None
            else predictions_by_pair.keys()
        )
        gt_by_pair = self.loader.load_ground_truth(pair_keys, dataset_name)
        ordered_keys = [key for key in pair_keys if key in gt_by_pair]
        if not ordered_keys:
            logger.warning(
                "No ground truth found for method=%s dataset=%s.",
                method_name,
                dataset_name,
            )
            return None

        y_true = np.asarray([gt_by_pair[key] for key in ordered_keys], dtype=int)
        predictions = np.asarray(
            [predictions_by_pair[key] for key in ordered_keys], dtype=float
        )
        aligned_pair_keys_hash = compute_aligned_pair_keys_hash(ordered_keys)

        cv_result = self.evaluator.compute_cv(
            y_true=y_true,
            predictions=predictions,
            ci_level=ci_level,
            n_boot=n_boot,
        )
        logger.info(
            "Computed statistical-inference PAN metrics for %s/%s on %d samples: %s",
            dataset_name,
            method_name,
            len(ordered_keys),
            cv_result.metrics_mean,
        )

        return StatisticalInferenceMetrics(
            dataset_name=dataset_name,
            method_name=method_name,
            prediction_field=prediction_field,
            alpha=None,
            left_p_value_field=None,
            right_p_value_field=None,
            type_i_error=None,
            type_ii_error=None,
            n_different_author=None,
            n_same_author=None,
            n_samples=len(ordered_keys),
            aligned_pair_keys_hash=aligned_pair_keys_hash,
            split_config=cv_result.split_config,
            metrics_mean=cv_result.metrics_mean,
            metrics_std=cv_result.metrics_std,
            metrics_ci=cv_result.metrics_ci,
            metric_values=cv_result.metric_values,
        )

    def evaluate(
        self,
        *,
        datasets: Iterable[str],
        methods: Iterable[str],
        prediction_field: str,
        n_impostors: int,
        n_potential_impostors: int | None,
        ci_level: float,
        n_boot: int,
        require_all_methods: bool = False,
    ) -> list[StatisticalInferenceMetrics]:
        results: list[StatisticalInferenceMetrics] = []
        methods = list(methods)
        for dataset_name in datasets:
            predictions_by_method: (
                dict[str, dict[tuple[ObjectId, ObjectId], float]] | None
            ) = None
            common_pair_keys: set[tuple[ObjectId, ObjectId]] | None = None
            if require_all_methods:
                predictions_by_method = self.loader.load_predictions_for_methods(
                    methods,
                    dataset_name,
                    prediction_field=prediction_field,
                    n_impostors=n_impostors,
                    n_potential_impostors=n_potential_impostors,
                )
                method_pair_sets = [
                    set(predictions_by_pair)
                    for predictions_by_pair in predictions_by_method.values()
                    if predictions_by_pair
                ]
                if len(method_pair_sets) != len(predictions_by_method):
                    missing_methods = [
                        method_name
                        for method_name, predictions_by_pair in predictions_by_method.items()
                        if not predictions_by_pair
                    ]
                    logger.warning(
                        "Skipping dataset=%s prediction alignment because methods have no predictions: %s.",
                        dataset_name,
                        ", ".join(missing_methods),
                    )
                    continue
                common_pair_keys = set.intersection(*method_pair_sets)
                if not common_pair_keys:
                    logger.warning(
                        "Skipping dataset=%s prediction alignment because selected methods have no shared pairs.",
                        dataset_name,
                    )
                    continue
                logger.info(
                    "Restricting dataset=%s predictions to %d pair(s) shared by all %d selected methods.",
                    dataset_name,
                    len(common_pair_keys),
                    len(predictions_by_method),
                )

            for method_name in methods:
                result = self.evaluate_method(
                    method_name,
                    dataset_name=dataset_name,
                    prediction_field=prediction_field,
                    n_impostors=n_impostors,
                    n_potential_impostors=n_potential_impostors,
                    ci_level=ci_level,
                    n_boot=n_boot,
                    predictions_by_pair=(
                        None
                        if predictions_by_method is None
                        else predictions_by_method[method_name]
                    ),
                    required_pair_keys=common_pair_keys,
                )
                if result is not None:
                    results.append(result)
        return results

    def evaluate_method_alpha_sweep(
        self,
        method_name: str,
        *,
        dataset_name: str,
        alphas: Sequence[float],
        left_p_value_field: str,
        right_p_value_field: str,
        n_impostors: int,
        n_potential_impostors: int | None,
        ci_level: float,
        n_boot: int,
        p_values_by_pair: (
            dict[tuple[ObjectId, ObjectId], tuple[float, float]] | None
        ) = None,
        required_pair_keys: set[tuple[ObjectId, ObjectId]] | None = None,
    ) -> list[StatisticalInferenceMetrics]:
        if p_values_by_pair is None:
            p_values_by_pair = self.loader.load_directional_p_values(
                method_name,
                dataset_name,
                left_p_value_field=left_p_value_field,
                right_p_value_field=right_p_value_field,
                n_impostors=n_impostors,
                n_potential_impostors=n_potential_impostors,
            )
        if not p_values_by_pair:
            logger.warning(
                "No p-values found for alpha sweep method=%s dataset=%s fields=%s/%s.",
                method_name,
                dataset_name,
                left_p_value_field,
                right_p_value_field,
            )
            return []

        pair_keys = sorted(
            required_pair_keys
            if required_pair_keys is not None
            else p_values_by_pair.keys()
        )
        gt_by_pair = self.loader.load_ground_truth(pair_keys, dataset_name)
        ordered_keys = [key for key in pair_keys if key in gt_by_pair]
        if not ordered_keys:
            logger.warning(
                "No ground truth found for alpha sweep method=%s dataset=%s.",
                method_name,
                dataset_name,
            )
            return []

        y_true = np.asarray([gt_by_pair[key] for key in ordered_keys], dtype=int)
        left_p_values = np.asarray(
            [p_values_by_pair[key][0] for key in ordered_keys],
            dtype=float,
        )
        right_p_values = np.asarray(
            [p_values_by_pair[key][1] for key in ordered_keys],
            dtype=float,
        )
        aligned_pair_keys_hash = compute_aligned_pair_keys_hash(ordered_keys)
        splits = self.evaluator.split_manager.build_splits(y_true)

        results: list[StatisticalInferenceMetrics] = []
        for alpha in alphas:
            predictions = ((left_p_values <= alpha) & (right_p_values <= alpha)).astype(
                float
            )
            error_rates = _classification_error_rates(y_true, predictions)
            cv_result = self.evaluator.compute_cv(
                y_true=y_true,
                predictions=predictions,
                ci_level=ci_level,
                n_boot=n_boot,
                splits=splits,
            )
            results.append(
                StatisticalInferenceMetrics(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    prediction_field=_prediction_field_for_alpha(alpha),
                    alpha=float(alpha),
                    left_p_value_field=left_p_value_field,
                    right_p_value_field=right_p_value_field,
                    type_i_error=_float_or_none(error_rates["type_i_error"]),
                    type_ii_error=_float_or_none(error_rates["type_ii_error"]),
                    n_different_author=int(error_rates["n_different_author"]),
                    n_same_author=int(error_rates["n_same_author"]),
                    n_samples=len(ordered_keys),
                    aligned_pair_keys_hash=aligned_pair_keys_hash,
                    split_config=cv_result.split_config,
                    metrics_mean=cv_result.metrics_mean,
                    metrics_std=cv_result.metrics_std,
                    metrics_ci=cv_result.metrics_ci,
                    metric_values=cv_result.metric_values,
                )
            )

        logger.info(
            "Computed alpha sweep for %s/%s on %d samples and %d alpha values.",
            dataset_name,
            method_name,
            len(ordered_keys),
            len(results),
        )
        return results

    def evaluate_alpha_sweep(
        self,
        *,
        datasets: Iterable[str],
        methods: Iterable[str],
        alphas: Sequence[float],
        left_p_value_field: str,
        right_p_value_field: str,
        n_impostors: int,
        n_potential_impostors: int | None,
        ci_level: float,
        n_boot: int,
        require_all_methods: bool = False,
    ) -> list[StatisticalInferenceMetrics]:
        results: list[StatisticalInferenceMetrics] = []
        for dataset_name in datasets:
            p_values_by_method: (
                dict[str, dict[tuple[ObjectId, ObjectId], tuple[float, float]]] | None
            ) = None
            common_pair_keys: set[tuple[ObjectId, ObjectId]] | None = None
            if require_all_methods:
                logger.info("Run on shared pairs of dataset=%s.", dataset_name)
                p_values_by_method = self.loader.load_directional_p_values_for_methods(
                    methods,
                    dataset_name,
                    left_p_value_field=left_p_value_field,
                    right_p_value_field=right_p_value_field,
                    n_impostors=n_impostors,
                    n_potential_impostors=n_potential_impostors,
                )
                method_pair_sets = [
                    set(p_values_by_pair)
                    for p_values_by_pair in p_values_by_method.values()
                    if p_values_by_pair
                ]
                if len(method_pair_sets) != len(p_values_by_method):
                    missing_methods = [
                        method_name
                        for method_name, p_values_by_pair in p_values_by_method.items()
                        if not p_values_by_pair
                    ]
                    logger.warning(
                        "Skipping dataset=%s alpha sweep alignment because methods have no p-values: %s.",
                        dataset_name,
                        ", ".join(missing_methods),
                    )
                    continue
                common_pair_keys = set.intersection(*method_pair_sets)
                if not common_pair_keys:
                    logger.warning(
                        "Skipping dataset=%s alpha sweep alignment because selected methods have no shared pairs.",
                        dataset_name,
                    )
                    continue
                logger.info(
                    "Restricting dataset=%s alpha sweep to %d pair(s) shared by all %d selected methods.",
                    dataset_name,
                    len(common_pair_keys),
                    len(p_values_by_method),
                )

            for method_name in methods:
                results.extend(
                    self.evaluate_method_alpha_sweep(
                        method_name,
                        dataset_name=dataset_name,
                        alphas=alphas,
                        left_p_value_field=left_p_value_field,
                        right_p_value_field=right_p_value_field,
                        n_impostors=n_impostors,
                        n_potential_impostors=n_potential_impostors,
                        ci_level=ci_level,
                        n_boot=n_boot,
                        p_values_by_pair=(
                            None
                            if p_values_by_method is None
                            else p_values_by_method[method_name]
                        ),
                        required_pair_keys=common_pair_keys,
                    )
                )
        return results

    def evaluate_directional_agreement_method(
        self,
        method_name: str,
        *,
        dataset_name: str,
        right_prediction_field: str = RIGHT_DISPUTED_LEFT_CANDIDATE_PREDICTION_FIELD,
        left_prediction_field: str = LEFT_DISPUTED_RIGHT_CANDIDATE_PREDICTION_FIELD,
        n_impostors: int,
        n_potential_impostors: int | None,
        right_predictions_by_pair: dict[tuple[ObjectId, ObjectId], float] | None = None,
        left_predictions_by_pair: dict[tuple[ObjectId, ObjectId], float] | None = None,
        required_pair_keys: set[tuple[ObjectId, ObjectId]] | None = None,
    ) -> StatisticalInferenceAgreement | None:
        if right_predictions_by_pair is None:
            right_predictions_by_pair = self.loader.load_predictions(
                method_name,
                dataset_name,
                prediction_field=right_prediction_field,
                n_impostors=n_impostors,
                n_potential_impostors=n_potential_impostors,
            )
        if left_predictions_by_pair is None:
            left_predictions_by_pair = self.loader.load_predictions(
                method_name,
                dataset_name,
                prediction_field=left_prediction_field,
                n_impostors=n_impostors,
                n_potential_impostors=n_potential_impostors,
            )
        ordered_keys = sorted(
            required_pair_keys
            if required_pair_keys is not None
            else set(right_predictions_by_pair) & set(left_predictions_by_pair)
        )
        if not ordered_keys:
            logger.warning(
                "No aligned directional predictions found for method=%s dataset=%s fields=%s/%s.",
                method_name,
                dataset_name,
                right_prediction_field,
                left_prediction_field,
            )
            return None

        right_predictions = np.asarray(
            [right_predictions_by_pair[key] for key in ordered_keys],
            dtype=int,
        )
        left_predictions = np.asarray(
            [left_predictions_by_pair[key] for key in ordered_keys],
            dtype=int,
        )
        observed_agreement = float(np.mean(right_predictions == left_predictions))
        kappa = _float_or_none(
            cohen_kappa_score(right_predictions, left_predictions, labels=[0, 1])
        )

        logger.info(
            "Computed directional prediction agreement for %s/%s on %d samples: "
            "kappa=%s, agreement=%.3f",
            dataset_name,
            method_name,
            len(ordered_keys),
            "--" if kappa is None else f"{kappa:.3f}",
            observed_agreement,
        )

        return StatisticalInferenceAgreement(
            dataset_name=dataset_name,
            method_name=method_name,
            right_prediction_field=right_prediction_field,
            left_prediction_field=left_prediction_field,
            alpha=None,
            right_p_value_field=None,
            left_p_value_field=None,
            n_samples=len(ordered_keys),
            aligned_pair_keys_hash=compute_aligned_pair_keys_hash(ordered_keys),
            cohen_kappa=kappa,
            observed_agreement=observed_agreement,
        )

    def evaluate_directional_agreement(
        self,
        *,
        datasets: Iterable[str],
        methods: Iterable[str],
        right_prediction_field: str = RIGHT_DISPUTED_LEFT_CANDIDATE_PREDICTION_FIELD,
        left_prediction_field: str = LEFT_DISPUTED_RIGHT_CANDIDATE_PREDICTION_FIELD,
        n_impostors: int,
        n_potential_impostors: int | None,
        require_all_methods: bool = False,
    ) -> list[StatisticalInferenceAgreement]:
        results: list[StatisticalInferenceAgreement] = []
        methods = list(methods)
        for dataset_name in datasets:
            right_predictions_by_method: (
                dict[str, dict[tuple[ObjectId, ObjectId], float]] | None
            ) = None
            left_predictions_by_method: (
                dict[str, dict[tuple[ObjectId, ObjectId], float]] | None
            ) = None
            common_pair_keys: set[tuple[ObjectId, ObjectId]] | None = None
            if require_all_methods:
                right_predictions_by_method = self.loader.load_predictions_for_methods(
                    methods,
                    dataset_name,
                    prediction_field=right_prediction_field,
                    n_impostors=n_impostors,
                    n_potential_impostors=n_potential_impostors,
                )
                left_predictions_by_method = self.loader.load_predictions_for_methods(
                    methods,
                    dataset_name,
                    prediction_field=left_prediction_field,
                    n_impostors=n_impostors,
                    n_potential_impostors=n_potential_impostors,
                )
                method_pair_sets = [
                    set(right_predictions_by_method[method_name])
                    & set(left_predictions_by_method[method_name])
                    for method_name in methods
                    if right_predictions_by_method[method_name]
                    and left_predictions_by_method[method_name]
                ]
                if len(method_pair_sets) != len(methods):
                    missing_methods = [
                        method_name
                        for method_name in methods
                        if not right_predictions_by_method[method_name]
                        or not left_predictions_by_method[method_name]
                    ]
                    logger.warning(
                        "Skipping dataset=%s directional prediction alignment because methods have no aligned predictions: %s.",
                        dataset_name,
                        ", ".join(missing_methods),
                    )
                    continue
                common_pair_keys = set.intersection(*method_pair_sets)
                if not common_pair_keys:
                    logger.warning(
                        "Skipping dataset=%s directional prediction alignment because selected methods have no shared pairs.",
                        dataset_name,
                    )
                    continue

            for method_name in methods:
                result = self.evaluate_directional_agreement_method(
                    method_name,
                    dataset_name=dataset_name,
                    right_prediction_field=right_prediction_field,
                    left_prediction_field=left_prediction_field,
                    n_impostors=n_impostors,
                    n_potential_impostors=n_potential_impostors,
                    right_predictions_by_pair=(
                        None
                        if right_predictions_by_method is None
                        else right_predictions_by_method[method_name]
                    ),
                    left_predictions_by_pair=(
                        None
                        if left_predictions_by_method is None
                        else left_predictions_by_method[method_name]
                    ),
                    required_pair_keys=common_pair_keys,
                )
                if result is not None:
                    results.append(result)
        return results

    def evaluate_directional_agreement_method_alpha_sweep(
        self,
        method_name: str,
        *,
        dataset_name: str,
        alphas: Sequence[float],
        left_p_value_field: str,
        right_p_value_field: str,
        n_impostors: int,
        n_potential_impostors: int | None,
        p_values_by_pair: (
            dict[tuple[ObjectId, ObjectId], tuple[float, float]] | None
        ) = None,
        required_pair_keys: set[tuple[ObjectId, ObjectId]] | None = None,
    ) -> list[StatisticalInferenceAgreement]:
        if p_values_by_pair is None:
            p_values_by_pair = self.loader.load_directional_p_values(
                method_name,
                dataset_name,
                left_p_value_field=left_p_value_field,
                right_p_value_field=right_p_value_field,
                n_impostors=n_impostors,
                n_potential_impostors=n_potential_impostors,
            )
        ordered_keys = sorted(
            required_pair_keys if required_pair_keys is not None else p_values_by_pair
        )
        if not ordered_keys:
            logger.warning(
                "No p-values found for alpha-sweep directional agreement method=%s dataset=%s.",
                method_name,
                dataset_name,
            )
            return []

        left_p_values = np.asarray(
            [p_values_by_pair[key][0] for key in ordered_keys],
            dtype=float,
        )
        right_p_values = np.asarray(
            [p_values_by_pair[key][1] for key in ordered_keys],
            dtype=float,
        )
        aligned_pair_keys_hash = compute_aligned_pair_keys_hash(ordered_keys)

        results: list[StatisticalInferenceAgreement] = []
        for alpha in alphas:
            left_predictions = (left_p_values <= alpha).astype(int)
            right_predictions = (right_p_values <= alpha).astype(int)
            observed_agreement = float(np.mean(right_predictions == left_predictions))
            kappa = _float_or_none(
                cohen_kappa_score(right_predictions, left_predictions, labels=[0, 1])
            )
            results.append(
                StatisticalInferenceAgreement(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    right_prediction_field=(
                        f"{right_p_value_field}_pred_{_alpha_slug(alpha)}"
                    ),
                    left_prediction_field=(
                        f"{left_p_value_field}_pred_{_alpha_slug(alpha)}"
                    ),
                    alpha=float(alpha),
                    right_p_value_field=right_p_value_field,
                    left_p_value_field=left_p_value_field,
                    n_samples=len(ordered_keys),
                    aligned_pair_keys_hash=aligned_pair_keys_hash,
                    cohen_kappa=kappa,
                    observed_agreement=observed_agreement,
                )
            )

        logger.info(
            "Computed directional agreement alpha sweep for %s/%s on %d samples and %d alpha values.",
            dataset_name,
            method_name,
            len(ordered_keys),
            len(results),
        )
        return results

    def evaluate_directional_agreement_alpha_sweep(
        self,
        *,
        datasets: Iterable[str],
        methods: Iterable[str],
        alphas: Sequence[float],
        left_p_value_field: str,
        right_p_value_field: str,
        n_impostors: int,
        n_potential_impostors: int | None,
        require_all_methods: bool = False,
    ) -> list[StatisticalInferenceAgreement]:
        results: list[StatisticalInferenceAgreement] = []
        for dataset_name in datasets:
            p_values_by_method: (
                dict[str, dict[tuple[ObjectId, ObjectId], tuple[float, float]]] | None
            ) = None
            common_pair_keys: set[tuple[ObjectId, ObjectId]] | None = None
            if require_all_methods:
                p_values_by_method = self.loader.load_directional_p_values_for_methods(
                    methods,
                    dataset_name,
                    left_p_value_field=left_p_value_field,
                    right_p_value_field=right_p_value_field,
                    n_impostors=n_impostors,
                    n_potential_impostors=n_potential_impostors,
                )
                method_pair_sets = [
                    set(p_values_by_pair)
                    for p_values_by_pair in p_values_by_method.values()
                    if p_values_by_pair
                ]
                if len(method_pair_sets) != len(p_values_by_method):
                    missing_methods = [
                        method_name
                        for method_name, p_values_by_pair in p_values_by_method.items()
                        if not p_values_by_pair
                    ]
                    logger.warning(
                        "Skipping dataset=%s directional agreement alignment because methods have no p-values: %s.",
                        dataset_name,
                        ", ".join(missing_methods),
                    )
                    continue
                common_pair_keys = set.intersection(*method_pair_sets)
                if not common_pair_keys:
                    logger.warning(
                        "Skipping dataset=%s directional agreement alignment because selected methods have no shared pairs.",
                        dataset_name,
                    )
                    continue

            for method_name in methods:
                results.extend(
                    self.evaluate_directional_agreement_method_alpha_sweep(
                        method_name,
                        dataset_name=dataset_name,
                        alphas=alphas,
                        left_p_value_field=left_p_value_field,
                        right_p_value_field=right_p_value_field,
                        n_impostors=n_impostors,
                        n_potential_impostors=n_potential_impostors,
                        p_values_by_pair=(
                            None
                            if p_values_by_method is None
                            else p_values_by_method[method_name]
                        ),
                        required_pair_keys=common_pair_keys,
                    )
                )
        return results


def metrics_to_rows(
    results: Sequence[StatisticalInferenceMetrics],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "dataset_name": result.dataset_name,
            "dataset_label": _display_dataset(result.dataset_name),
            "method_name": result.method_name,
            "method_label": _display_method(result.method_name),
            "prediction_field": result.prediction_field,
            "alpha": result.alpha,
            "left_p_value_field": result.left_p_value_field,
            "right_p_value_field": result.right_p_value_field,
            "type_i_error": result.type_i_error,
            "type_ii_error": result.type_ii_error,
            "n_different_author": result.n_different_author,
            "n_same_author": result.n_same_author,
            "n_samples": result.n_samples,
            "aligned_pair_keys_hash": result.aligned_pair_keys_hash,
            "n_splits": result.split_config.get("n_splits"),
            "n_repeats": result.split_config.get("n_repeats"),
            "random_state": result.split_config.get("random_state"),
        }
        for metric in METRIC_ORDER:
            row[metric] = result.metrics_mean.get(metric)
            row[f"{metric}_std"] = result.metrics_std.get(metric)
            ci = result.metrics_ci.get(metric, {})
            row[f"{metric}_ci_low"] = ci.get("low")
            row[f"{metric}_ci_high"] = ci.get("high")
        rows.append(row)
    return rows


def write_csv(results: Sequence[StatisticalInferenceMetrics], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = metrics_to_rows(results)
    fieldnames = [
        "dataset_name",
        "dataset_label",
        "method_name",
        "method_label",
        "prediction_field",
        "alpha",
        "left_p_value_field",
        "right_p_value_field",
        "type_i_error",
        "type_ii_error",
        "n_different_author",
        "n_same_author",
        "n_samples",
        "aligned_pair_keys_hash",
        "n_splits",
        "n_repeats",
        "random_state",
    ]
    for metric in METRIC_ORDER:
        fieldnames.extend(
            [metric, f"{metric}_std", f"{metric}_ci_low", f"{metric}_ci_high"]
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def best_alpha_rows(
    results: Sequence[StatisticalInferenceMetrics],
    *,
    latex_paths_by_alpha_slug: dict[str, Path],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[StatisticalInferenceMetrics]] = defaultdict(
        list
    )
    for result in results:
        if result.alpha is not None:
            grouped[(result.dataset_name, result.method_name)].append(result)

    rows: list[dict[str, Any]] = []
    for (dataset_name, method_name), group_results in grouped.items():
        for metric in METRIC_ORDER:
            candidates = [
                (result.alpha, _float_or_none(result.metrics_mean.get(metric)))
                for result in group_results
            ]
            candidates = [
                (float(alpha), float(value))
                for alpha, value in candidates
                if alpha is not None and value is not None
            ]
            if not candidates:
                continue
            best_value = max(value for _, value in candidates)
            tied_alphas = sorted(
                alpha
                for alpha, value in candidates
                if math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12)
            )
            best_alpha = tied_alphas[0]
            best_result = next(
                result
                for result in group_results
                if result.alpha is not None
                and math.isclose(result.alpha, best_alpha, rel_tol=1e-12, abs_tol=1e-12)
            )
            best_alpha_slug = _alpha_slug(best_alpha)
            latex_path = latex_paths_by_alpha_slug.get(best_alpha_slug)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_label": _display_dataset(dataset_name),
                    "method_name": method_name,
                    "method_label": _display_method(method_name),
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "best_alpha": _format_alpha(best_alpha),
                    "best_value": best_value,
                    "type_i_error_at_best_alpha": best_result.type_i_error,
                    "type_ii_error_at_best_alpha": best_result.type_ii_error,
                    "n_different_author": best_result.n_different_author,
                    "n_same_author": best_result.n_same_author,
                    "n_tied_best_alphas": len(tied_alphas),
                    "tied_best_alphas": ";".join(
                        _format_alpha(alpha) for alpha in tied_alphas
                    ),
                    "latex_file": "" if latex_path is None else latex_path.name,
                    "latex_path": "" if latex_path is None else str(latex_path),
                }
            )
    return rows


def write_best_alpha_csv(
    results: Sequence[StatisticalInferenceMetrics],
    out_path: Path,
    *,
    latex_paths_by_alpha_slug: dict[str, Path],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = best_alpha_rows(
        results,
        latex_paths_by_alpha_slug=latex_paths_by_alpha_slug,
    )
    fieldnames = [
        "dataset_name",
        "dataset_label",
        "method_name",
        "method_label",
        "metric",
        "metric_label",
        "best_alpha",
        "best_value",
        "type_i_error_at_best_alpha",
        "type_ii_error_at_best_alpha",
        "n_different_author",
        "n_same_author",
        "n_tied_best_alphas",
        "tied_best_alphas",
        "latex_file",
        "latex_path",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _metric_maxima(results: Sequence[StatisticalInferenceMetrics]) -> dict[str, float]:
    maxima: dict[str, float] = {}
    for metric in METRIC_ORDER:
        values = [
            value
            for result in results
            if (value := _float_or_none(result.metrics_mean.get(metric))) is not None
        ]
        if values:
            maxima[metric] = max(values)
    return maxima


def build_latex_table(
    results: Sequence[StatisticalInferenceMetrics],
    *,
    caption: str,
    label: str,
    bold_max: bool = True,
) -> str:
    by_dataset: dict[str, list[StatisticalInferenceMetrics]] = defaultdict(list)
    for result in results:
        by_dataset[result.dataset_name].append(result)

    lines = [
        "% Auto-generated by genai_detection.experiments.evaluate_statistical_inference.",
        r"\begin{table}[htb]",
        r"\centering",
        r"\small",
        r"\fontsize{7.5pt}{8pt}\selectfont",
        r"\renewcommand{\tabcolsep}{2pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r" & & \multicolumn{6}{@{}c@{}}{\textbf{Reproduction Scores}} \\",
        r"\cmidrule(l@{\tabcolsep}){2-7}",
        " Method & "
        + " & ".join(METRIC_LABELS[metric] for metric in METRIC_ORDER)
        + r" \\",
        r"\midrule",
    ]

    dataset_items = list(by_dataset.items())
    for dataset_idx, (dataset_name, dataset_results) in enumerate(dataset_items):
        maxima = _metric_maxima(dataset_results) if bold_max else {}
        dataset_label = _latex_escape(
            _display_statistical_inference_dataset(dataset_name)
        )
        lines.append(
            rf"\multicolumn{{8}}{{@{{}}c@{{}}}}{{\emph{{{dataset_label}}}}}  \\"
        )
        for result in dataset_results:
            method_cell = _latex_escape(_display_method(result.method_name))
            metric_cells = []
            for metric in METRIC_ORDER:
                value = _float_or_none(result.metrics_mean.get(metric))
                is_bold = bold_max and value is not None and value == maxima.get(metric)
                metric_cells.append(_format_metric(value, bold=is_bold))
            lines.append(f"{method_cell} & " + " & ".join(metric_cells) + r" \\")
        if dataset_idx < len(dataset_items) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_latex_table(
    results: Sequence[StatisticalInferenceMetrics],
    out_path: Path,
    *,
    caption: str,
    label: str,
    bold_max: bool,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latex = build_latex_table(
        results,
        caption=caption,
        label=label,
        bold_max=bold_max,
    )
    out_path.write_text(latex, encoding="utf-8")
    return out_path


def build_kappa_latex_table(
    results: Sequence[StatisticalInferenceAgreement],
    *,
    caption: str,
    label: str,
) -> str:
    by_dataset: dict[str, list[StatisticalInferenceAgreement]] = defaultdict(list)
    for result in results:
        by_dataset[result.dataset_name].append(result)

    lines = [
        "% Auto-generated by genai_detection.experiments.evaluate_statistical_inference.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Dataset & Method & $n$ & Cohen's $\kappa$ \\",
        r"\midrule",
    ]

    dataset_items = list(by_dataset.items())
    for dataset_idx, (dataset_name, dataset_results) in enumerate(dataset_items):
        for method_idx, result in enumerate(dataset_results):
            dataset_cell = (
                _latex_escape(_display_dataset(dataset_name)) if method_idx == 0 else ""
            )
            method_cell = _latex_escape(_display_method(result.method_name))
            lines.append(
                f"{dataset_cell} & {method_cell} & {result.n_samples} & "
                f"{_format_metric(result.cohen_kappa)}" + r" \\"
            )
        if dataset_idx < len(dataset_items) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_kappa_latex_table(
    results: Sequence[StatisticalInferenceAgreement],
    out_path: Path,
    *,
    caption: str,
    label: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latex = build_kappa_latex_table(
        results,
        caption=caption,
        label=label,
    )
    out_path.write_text(latex, encoding="utf-8")
    return out_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    output_dir = _default_output_dir()
    safe_field = _safe_filename(DEFAULT_PREDICTION_FIELD)

    parser = argparse.ArgumentParser(
        description=(
            "Compute PAN metrics for binary statistical-inference predictions "
            "stored in MongoDB impostors_outputs."
        )
    )
    parser.add_argument(
        "--datasets", nargs="+", default=[CONFIG.BLOG, CONFIG.STUDENT_ESSAYS]
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHOD_ORDER))
    parser.add_argument("--prediction-field", default=DEFAULT_PREDICTION_FIELD)
    parser.add_argument("--n-impostors", type=int, default=N_IMPOSTORS)
    parser.add_argument(
        "--n-potential-impostors", type=int, default=N_POTENTIAL_IMPOSTORS
    )
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--ci-level", type=float, default=CI_LEVEL)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--alpha-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate alpha-specific outputs from stored p-values. Enabled by default.",
    )
    parser.add_argument(
        "--left-p-value-field",
        default=LEFT_DISPUTED_RIGHT_CANDIDATE_CORRECTED_P_VALUE_FIELD,
        help="MongoDB field for left-disputed/right-candidate p-values used in the alpha sweep.",
    )
    parser.add_argument(
        "--right-p-value-field",
        default=RIGHT_DISPUTED_LEFT_CANDIDATE_CORRECTED_P_VALUE_FIELD,
        help="MongoDB field for right-disputed/left-candidate p-values used in the alpha sweep.",
    )
    parser.add_argument(
        "--alpha-sweep-dir",
        type=Path,
        default=None,
        help="Directory for alpha-specific TeX and CSV files. Defaults to output-dir/alpha_sweeps.",
    )
    parser.add_argument(
        "--alpha-sweep-csv-output",
        type=Path,
        default=None,
        help="Combined CSV for all alpha-sweep PAN metrics.",
    )
    parser.add_argument(
        "--best-alpha-output",
        type=Path,
        default=None,
        help="CSV containing the best alpha for each dataset/method/metric.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=output_dir / f"pan_metrics_statistical_inference_{safe_field}.csv",
    )
    parser.add_argument(
        "--latex", "--latex-table", action="store_true", dest="latex", default=True
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=output_dir / f"pan_metrics_statistical_inference_{safe_field}.tex",
    )
    parser.add_argument(
        "--latex-caption",
        default=(
            "PAN metrics for statistical-inference formulation of the Impostors Method "
            r"($\alpha=0.05$). ChatNoir and Startpage are retrieval-based approaches."
        ),
    )
    parser.add_argument(
        "--alpha-sweep-latex-caption",
        default=(
            "PAN metrics for statistical-inference formulation of the Impostors Method "
            r"($\alpha={alpha}$). ChatNoir and Startpage are retrieval-based approaches."
        ),
    )
    parser.add_argument(
        "--latex-label",
        default="tab:statistical-inference-impostor-pan-metrics",
    )
    parser.add_argument(
        "--kappa-latex-output",
        type=Path,
        default=output_dir / KAPPA_LATEX_OUTPUT_FILENAME,
    )
    parser.add_argument(
        "--kappa-latex-caption",
        default=(
            r"Cohen's $\kappa$ for agreement between the two directional statistical-inference decisions; "
            r"$n$ denotes the number of aligned pairs."
        ),
    )
    parser.add_argument(
        "--kappa-alpha-sweep-latex-caption",
        default=(
            r"Cohen's $\kappa$ for agreement between the two directional statistical-inference decisions "
            r"at $\alpha={alpha}$; $n$ denotes the number of aligned pairs."
        ),
    )
    parser.add_argument(
        "--kappa-latex-label",
        default="tab:statistical-inference-directional-kappa-agreement",
    )
    parser.add_argument("--no-bold-max", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))

    if (
        args.csv_output
        == _default_output_dir()
        / f"pan_metrics_statistical_inference_{_safe_filename(DEFAULT_PREDICTION_FIELD)}.csv"
    ):
        args.csv_output = (
            args.output_dir
            / f"pan_metrics_statistical_inference_{_safe_filename(args.prediction_field)}.csv"
        )
    if (
        args.latex_output
        == _default_output_dir()
        / f"pan_metrics_statistical_inference_{_safe_filename(DEFAULT_PREDICTION_FIELD)}.tex"
    ):
        args.latex_output = (
            args.output_dir
            / f"pan_metrics_statistical_inference_{_safe_filename(args.prediction_field)}.tex"
        )
    if args.kappa_latex_output == _default_output_dir() / KAPPA_LATEX_OUTPUT_FILENAME:
        args.kappa_latex_output = args.output_dir / KAPPA_LATEX_OUTPUT_FILENAME
    if args.alpha_sweep_dir is None:
        args.alpha_sweep_dir = args.output_dir / ALPHA_SWEEP_DIRNAME
    if args.alpha_sweep_csv_output is None:
        args.alpha_sweep_csv_output = (
            args.alpha_sweep_dir
            / f"pan_metrics_statistical_inference_{_safe_filename(args.prediction_field)}_alpha_sweep.csv"
        )
    if args.best_alpha_output is None:
        args.best_alpha_output = (
            args.alpha_sweep_dir / "best_alpha_by_metric_method_dataset.csv"
        )

    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    loader = StatisticalInferencePredictionLoader(
        mongo=mongo, batch_size=args.batch_size
    )
    split_manager = SplitManager(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    evaluator = FixedPredictionPANEvaluator(
        metric_computer=PANMetricComputer(),
        split_manager=split_manager,
    )
    experiment = StatisticalInferenceExperiment(loader=loader, evaluator=evaluator)

    # True -> shared subset of pairs across all methods for a dataset
    for require_all_methods in [False, True]:
        csv_output = _with_shared_subset_suffix(args.csv_output, require_all_methods)
        latex_output = _with_shared_subset_suffix(
            args.latex_output, require_all_methods
        )
        kappa_latex_output = _with_shared_subset_suffix(
            args.kappa_latex_output, require_all_methods
        )
        alpha_sweep_csv_output = _with_shared_subset_suffix(
            args.alpha_sweep_csv_output, require_all_methods
        )
        best_alpha_output = _with_shared_subset_suffix(
            args.best_alpha_output, require_all_methods
        )

        if args.alpha_sweep:
            alphas = _alpha_grid(DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX, DEFAULT_N_ALPHA)
            logger.info(
                "Running alpha sweep with %d values: %s to %s with %s alpha values on shared subset of data: %s",
                len(alphas),
                _format_alpha(alphas[0]),
                _format_alpha(alphas[-1]),
                DEFAULT_N_ALPHA,
                require_all_methods,
            )
            if not math.isclose(alphas[-1], DEFAULT_ALPHA_MAX, rel_tol=0.0, abs_tol=1e-12):
                logger.info(
                    "Alpha max %.12g is not on the requested grid; last evaluated alpha is %s.",
                    DEFAULT_ALPHA_MAX,
                    _format_alpha(alphas[-1]),
                )

            results = experiment.evaluate_alpha_sweep(
                datasets=args.datasets,
                methods=args.methods,
                alphas=alphas,
                left_p_value_field=args.left_p_value_field,
                right_p_value_field=args.right_p_value_field,
                n_impostors=args.n_impostors,
                n_potential_impostors=args.n_potential_impostors,
                ci_level=args.ci_level,
                n_boot=args.n_boot,
                require_all_methods=require_all_methods,
            )
            agreement_results = experiment.evaluate_directional_agreement_alpha_sweep(
                datasets=args.datasets,
                methods=args.methods,
                alphas=alphas,
                left_p_value_field=args.left_p_value_field,
                right_p_value_field=args.right_p_value_field,
                n_impostors=args.n_impostors,
                n_potential_impostors=args.n_potential_impostors,
                require_all_methods=require_all_methods,
            )

            if not results and not agreement_results:
                raise SystemExit(
                    "No alpha-sweep statistical-inference metrics or agreement results were computed."
                )

            latex_paths_by_alpha_slug: dict[str, Path] = {}
            if results:
                csv_path = write_csv(results, alpha_sweep_csv_output)
                logger.info(
                    "Wrote alpha-sweep PAN metrics CSV to %s. On shared subset: %s",
                    csv_path,
                    require_all_methods,
                )
                plot_results(
                    csv_path=csv_path,
                    require_all_methods=require_all_methods,
                )
            else:
                logger.warning("No alpha-sweep PAN metrics were computed.")

            if args.latex and results:
                results_by_alpha_slug: dict[str, list[StatisticalInferenceMetrics]] = (
                    defaultdict(list)
                )
                for result in results:
                    assert result.alpha is not None
                    results_by_alpha_slug[_alpha_slug(result.alpha)].append(result)
                for alpha in alphas:
                    alpha_slug = _alpha_slug(alpha)
                    alpha_results = results_by_alpha_slug.get(alpha_slug)
                    if not alpha_results:
                        continue
                    latex_path = write_latex_table(
                        alpha_results,
                        _with_alpha_suffix(
                            latex_output, alpha, directory=args.alpha_sweep_dir
                        ),
                        caption=_caption_with_alpha(
                            args.alpha_sweep_latex_caption, alpha
                        ),
                        label=_label_with_alpha(args.latex_label, alpha),
                        bold_max=not args.no_bold_max,
                    )
                    latex_paths_by_alpha_slug[alpha_slug] = latex_path
                logger.info(
                    "Wrote %d alpha-specific statistical-inference PAN metrics LaTeX tables to %s.",
                    len(latex_paths_by_alpha_slug),
                    args.alpha_sweep_dir,
                )

            if results:
                best_alpha_path = write_best_alpha_csv(
                    results,
                    best_alpha_output,
                    latex_paths_by_alpha_slug=latex_paths_by_alpha_slug,
                )
                logger.info("Wrote best-alpha index CSV to %s.", best_alpha_path)

            if agreement_results:
                agreement_by_alpha_slug: dict[
                    str, list[StatisticalInferenceAgreement]
                ] = defaultdict(list)
                for result in agreement_results:
                    assert result.alpha is not None
                    agreement_by_alpha_slug[_alpha_slug(result.alpha)].append(result)
                written_kappa_paths = []
                for alpha in alphas:
                    alpha_slug = _alpha_slug(alpha)
                    alpha_agreement_results = agreement_by_alpha_slug.get(alpha_slug)
                    if not alpha_agreement_results:
                        continue
                    kappa_latex_path = write_kappa_latex_table(
                        alpha_agreement_results,
                        _with_alpha_suffix(
                            kappa_latex_output, alpha, directory=args.alpha_sweep_dir
                        ),
                        caption=_caption_with_alpha(
                            args.kappa_alpha_sweep_latex_caption, alpha
                        ),
                        label=_label_with_alpha(args.kappa_latex_label, alpha),
                    )
                    written_kappa_paths.append(kappa_latex_path)
                logger.info(
                    "Wrote %d alpha-specific directional prediction kappa LaTeX tables to %s.",
                    len(written_kappa_paths),
                    args.alpha_sweep_dir,
                )
            else:
                logger.warning(
                    "No alpha-sweep directional prediction agreement results were computed."
                )
            continue

        results = experiment.evaluate(
            datasets=args.datasets,
            methods=args.methods,
            prediction_field=args.prediction_field,
            n_impostors=args.n_impostors,
            n_potential_impostors=args.n_potential_impostors,
            ci_level=args.ci_level,
            n_boot=args.n_boot,
            require_all_methods=require_all_methods,
        )
        agreement_results = experiment.evaluate_directional_agreement(
            datasets=args.datasets,
            methods=args.methods,
            n_impostors=args.n_impostors,
            n_potential_impostors=args.n_potential_impostors,
            require_all_methods=require_all_methods,
        )

        if not results and not agreement_results:
            raise SystemExit(
                "No statistical-inference metrics or agreement results were computed."
            )

        if results:
            csv_path = write_csv(results, csv_output)
            logger.info("Wrote statistical-inference PAN metrics CSV to %s.", csv_path)
        else:
            logger.warning("No statistical-inference PAN metrics were computed.")

        if args.latex and results:
            latex_path = write_latex_table(
                results,
                latex_output,
                caption=args.latex_caption,
                label=args.latex_label,
                bold_max=not args.no_bold_max,
            )
            logger.info(
                "Wrote statistical-inference PAN metrics LaTeX table to %s.", latex_path
            )

        if agreement_results:
            kappa_latex_path = write_kappa_latex_table(
                agreement_results,
                kappa_latex_output,
                caption=args.kappa_latex_caption,
                label=args.kappa_latex_label,
            )
            logger.info(
                "Wrote directional prediction kappa agreement LaTeX table to %s.",
                kappa_latex_path,
            )
        else:
            logger.warning("No directional prediction agreement results were computed.")


METRICS = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auroc",
    "c_at_1",
    "auroc_c_at_1",
]


def plot_results(csv_path, *, require_all_methods: bool = False):
    # Input/output
    assert os.path.exists(csv_path), f"CSV file {csv_path} does not exist."
    output_dir = Path(csv_path).parent / "alpha_scatter_plots"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)

    for dataset_name in sorted(df["dataset_name"].unique()):
        dataset_df = df[df["dataset_name"] == dataset_name]

        fig, axes = plt.subplots(
            nrows=2,
            ncols=4,
            figsize=(16, 8),
            constrained_layout=True,
        )
        axes = axes.flatten()

        for i, metric in enumerate(METRICS):
            ax = axes[i]

            for method_name, method_df in dataset_df.groupby("method_name"):
                ax.scatter(
                    method_df["alpha"],
                    method_df[metric],
                    label=CONFIG.LABEL_TRANSLATIONS.get(method_name, method_name),
                    color=CONFIG.LABEL_COLORS.get(method_name),
                    s=10,
                )

            metric_label = CONFIG.SCORE_TRANSLATIONS.get(metric, metric)
            ax.set_title(metric_label)
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(metric_label)
            ax.set_ylim((0,1))
            ax.grid(alpha=0.3)

        # Remove unused subplot
        fig.delaxes(axes[-1])

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, -0.04),
        )

        dataset_label = CONFIG.DATASET_TRANSLATIONS.get(dataset_name, dataset_name)
        if require_all_methods:
            n_samples = sorted(dataset_df["n_samples"].dropna().unique())
            if len(n_samples) == 1:
                dataset_label = f"{dataset_label} (n={int(n_samples[0])})"
        fig.suptitle(dataset_label, fontsize=16)
        for type in ["png", "svg", "pdf"]:
            plot_path = (
                output_dir
                / f"{dataset_name}_alpha_scatter_shared_subset_{require_all_methods}.{type}"
            )
            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)
        logger.info(
            "Saved scatter plot for %s to %s",
            dataset_name,
            output_dir
            / f"{dataset_name}_alpha_scatter_shared_subset_{require_all_methods}.svg",
        )


if __name__ == "__main__":
    main()
