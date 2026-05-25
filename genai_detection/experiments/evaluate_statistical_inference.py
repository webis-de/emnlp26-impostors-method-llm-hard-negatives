from __future__ import annotations

"""Evaluate statistical-inference impostor predictions with PAN metrics."""

import argparse
import csv
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from bson import ObjectId
from sklearn.metrics import cohen_kappa_score

from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.experiments.reproduction.pan_metrics.pan_cv import (
    SplitManager,
    compute_aligned_pair_keys_hash,
)
from genai_detection.experiments.reproduction.pan_metrics.pan_data_loader import PANDataLoader
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

DEFAULT_PREDICTION_FIELD = "pred_both_hypothesis_directions"
RIGHT_DISPUTED_LEFT_CANDIDATE_PREDICTION_FIELD = (
    "right_disputed_left_candidate_uncorrected_p_value_pred"
)
LEFT_DISPUTED_RIGHT_CANDIDATE_PREDICTION_FIELD = (
    "left_disputed_right_candidate_uncorrected_p_value_pred"
)
KAPPA_LATEX_OUTPUT_FILENAME = (
    "kappa_annotator_agreement_directional_uncorrected_p_value_predictions.tex"
)
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

STATISTICAL_INFERENCE_METHOD_LABELS = {
    "on_the_fly_chatnoir": "ChatNoir",
    "on_the_fly_startpage": "Startpage",
    "in_domain": "In-Domain",
    "two_step_llm": "LLM impostors",
}

DEFAULT_METHOD_ORDER = [
    "on_the_fly_chatnoir",
    "on_the_fly_startpage",
    "in_domain",
    "two_step_llm",

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
    return STATISTICAL_INFERENCE_METHOD_LABELS.get(method_name, _display_method(method_name))


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

    def _load_impostor_predictions(
        self,
        technique: str,
        dataset_name: str,
        *,
        prediction_field: str,
        n_impostors: int,
        n_potential_impostors: int | None = None,
    ) -> dict[tuple[ObjectId, ObjectId], float]:
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

    def __init__(self, metric_computer: PANMetricComputer, split_manager: SplitManager) -> None:
        self.metric_computer = metric_computer
        self.split_manager = split_manager

    def compute_cv(
        self,
        y_true: Sequence[int] | np.ndarray,
        predictions: Sequence[float] | np.ndarray,
        *,
        ci_level: float = CI_LEVEL,
        n_boot: int = N_BOOT,
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

        splits = self.split_manager.build_splits(y_true_np)
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

        metrics_mean, metrics_std, metric_values, metrics_ci = PANMetricComputer._summarize_metrics(
            per_split_results,
            ci_level=ci_level,
            n_boot=n_boot,
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
    ) -> StatisticalInferenceMetrics | None:
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

        pair_keys = sorted(predictions_by_pair.keys())
        gt_by_pair = self.loader.load_ground_truth(pair_keys, dataset_name)
        ordered_keys = [key for key in pair_keys if key in gt_by_pair]
        if not ordered_keys:
            logger.warning("No ground truth found for method=%s dataset=%s.", method_name, dataset_name)
            return None

        y_true = np.asarray([gt_by_pair[key] for key in ordered_keys], dtype=int)
        predictions = np.asarray([predictions_by_pair[key] for key in ordered_keys], dtype=float)
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
    ) -> list[StatisticalInferenceMetrics]:
        results: list[StatisticalInferenceMetrics] = []
        for dataset_name in datasets:
            for method_name in methods:
                result = self.evaluate_method(
                    method_name,
                    dataset_name=dataset_name,
                    prediction_field=prediction_field,
                    n_impostors=n_impostors,
                    n_potential_impostors=n_potential_impostors,
                    ci_level=ci_level,
                    n_boot=n_boot,
                )
                if result is not None:
                    results.append(result)
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
    ) -> StatisticalInferenceAgreement | None:
        right_predictions_by_pair = self.loader.load_predictions(
            method_name,
            dataset_name,
            prediction_field=right_prediction_field,
            n_impostors=n_impostors,
            n_potential_impostors=n_potential_impostors,
        )
        left_predictions_by_pair = self.loader.load_predictions(
            method_name,
            dataset_name,
            prediction_field=left_prediction_field,
            n_impostors=n_impostors,
            n_potential_impostors=n_potential_impostors,
        )
        ordered_keys = sorted(
            set(right_predictions_by_pair) & set(left_predictions_by_pair)
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
    ) -> list[StatisticalInferenceAgreement]:
        results: list[StatisticalInferenceAgreement] = []
        for dataset_name in datasets:
            for method_name in methods:
                result = self.evaluate_directional_agreement_method(
                    method_name,
                    dataset_name=dataset_name,
                    right_prediction_field=right_prediction_field,
                    left_prediction_field=left_prediction_field,
                    n_impostors=n_impostors,
                    n_potential_impostors=n_potential_impostors,
                )
                if result is not None:
                    results.append(result)
        return results


def metrics_to_rows(results: Sequence[StatisticalInferenceMetrics]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "dataset_name": result.dataset_name,
            "dataset_label": _display_dataset(result.dataset_name),
            "method_name": result.method_name,
            "method_label": _display_method(result.method_name),
            "prediction_field": result.prediction_field,
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
        "n_samples",
        "aligned_pair_keys_hash",
        "n_splits",
        "n_repeats",
        "random_state",
    ]
    for metric in METRIC_ORDER:
        fieldnames.extend([metric, f"{metric}_std", f"{metric}_ci_low", f"{metric}_ci_high"])

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
        dataset_label = _latex_escape(_display_statistical_inference_dataset(dataset_name))
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
            lines.append(
                f"{method_cell} & " + " & ".join(metric_cells) + r" \\"
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
    parser.add_argument("--datasets", nargs="+", default=[CONFIG.BLOG, CONFIG.STUDENT_ESSAYS])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHOD_ORDER))
    parser.add_argument("--prediction-field", default=DEFAULT_PREDICTION_FIELD)
    parser.add_argument("--n-impostors", type=int, default=N_IMPOSTORS)
    parser.add_argument("--n-potential-impostors", type=int, default=N_POTENTIAL_IMPOSTORS)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--ci-level", type=float, default=CI_LEVEL)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=output_dir / f"pan_metrics_statistical_inference_{safe_field}.csv",
    )
    parser.add_argument("--latex", "--latex-table", action="store_true", dest="latex", default=True)
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
        "--kappa-latex-label",
        default="tab:statistical-inference-directional-kappa-agreement",
    )
    parser.add_argument("--no-bold-max", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))

    if args.csv_output == _default_output_dir() / f"pan_metrics_statistical_inference_{_safe_filename(DEFAULT_PREDICTION_FIELD)}.csv":
        args.csv_output = args.output_dir / f"pan_metrics_statistical_inference_{_safe_filename(args.prediction_field)}.csv"
    if args.latex_output == _default_output_dir() / f"pan_metrics_statistical_inference_{_safe_filename(DEFAULT_PREDICTION_FIELD)}.tex":
        args.latex_output = args.output_dir / f"pan_metrics_statistical_inference_{_safe_filename(args.prediction_field)}.tex"
    if args.kappa_latex_output == _default_output_dir() / KAPPA_LATEX_OUTPUT_FILENAME:
        args.kappa_latex_output = args.output_dir / KAPPA_LATEX_OUTPUT_FILENAME

    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    loader = StatisticalInferencePredictionLoader(mongo=mongo, batch_size=args.batch_size)
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

    results = experiment.evaluate(
        datasets=args.datasets,
        methods=args.methods,
        prediction_field=args.prediction_field,
        n_impostors=args.n_impostors,
        n_potential_impostors=args.n_potential_impostors,
        ci_level=args.ci_level,
        n_boot=args.n_boot,
    )
    agreement_results = experiment.evaluate_directional_agreement(
        datasets=args.datasets,
        methods=args.methods,
        n_impostors=args.n_impostors,
        n_potential_impostors=args.n_potential_impostors,
    )

    if not results and not agreement_results:
        raise SystemExit("No statistical-inference metrics or agreement results were computed.")

    if results:
        csv_path = write_csv(results, args.csv_output)
        logger.info("Wrote statistical-inference PAN metrics CSV to %s.", csv_path)
    else:
        logger.warning("No statistical-inference PAN metrics were computed.")

    if args.latex and results:
        latex_path = write_latex_table(
            results,
            args.latex_output,
            caption=args.latex_caption,
            label=args.latex_label,
            bold_max=not args.no_bold_max,
        )
        logger.info("Wrote statistical-inference PAN metrics LaTeX table to %s.", latex_path)

    if agreement_results:
        kappa_latex_path = write_kappa_latex_table(
            agreement_results,
            args.kappa_latex_output,
            caption=args.kappa_latex_caption,
            label=args.kappa_latex_label,
        )
        logger.info(
            "Wrote directional prediction kappa agreement LaTeX table to %s.",
            kappa_latex_path,
        )
    else:
        logger.warning("No directional prediction agreement results were computed.")


if __name__ == "__main__":
    main()
