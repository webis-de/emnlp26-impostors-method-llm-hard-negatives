"""Compute and store PAN metrics for selected methods."""

from __future__ import annotations

from dataclasses import asdict
import logging

import numpy as np

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pan_metrics.pan_cv import PANEvaluator, SplitManager
from genai_detection.experiments.reproduction.pan_metrics.pan_data_loader import PANDataLoader
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import PANMetricComputer
from genai_detection.experiments.reproduction.pan_metrics.pan_storage import PANMetricsRecord, PANMetricsStore

logger = logging.getLogger(__name__)


IMPOSTOR_TECHNIQUE_FOR_ABLATIONS = "in_domain"

# Comment out methods you don't want to run.
IMPOSTOR_METHODS = [
    "two_step_llm",
    "on_the_fly_chatnoir",
    # "on_the_fly_serpapi",
    "on_the_fly_startpage",
    "in_domain",
    "one_step_llm",
    # "translation",
    # "mirror_minds",   # raises error
]

ABLATION_METHODS = [
    "bdi",
    "homotopy",
    "potha2017",
    "asgalf",
    "std_impostor",
    "unmasking",
    "unsupervised_baseline_min-max",
    "unsupervised_baseline_cosine",
    "supervised_baseline",
    "ppmd",
]

N_IMPOSTORS = 50
N_POTENTIAL_IMPOSTORS = None
ROUNDS = 100

N_SPLITS = 10
N_REPEATS = 5
RANDOM_STATE = 42
CI_LEVEL = 0.95
N_BOOT = 10000


def _compute_for_method(
    method_name: str,
    *,
    dataset_name: str,
    loader: PANDataLoader,
    evaluator: PANEvaluator,
    store: PANMetricsStore,
    impostor_technique: str | None,
    n_impostors: int,
    n_potential_impostors: int | None,
    rounds: int,
) -> None:
    scores_by_pair = loader.load_scores(
        method_name,
        dataset_name,
        impostor_technique=impostor_technique,
        n_impostors=n_impostors,
        n_potential_impostors=n_potential_impostors,
        rounds=rounds,
    )
    if not scores_by_pair:
        logger.warning("No scores found for %s (dataset=%s).", method_name, dataset_name)
        return
    logger.info("Loaded %d scores for %s (dataset=%s).", len(scores_by_pair), method_name, dataset_name)

    pair_keys = sorted(scores_by_pair.keys())
    gt_by_pair = loader.load_ground_truth(pair_keys, dataset_name)
    ordered_keys = [key for key in pair_keys if key in gt_by_pair]
    if not ordered_keys:
        logger.warning("No ground truth available for %s (dataset=%s).", method_name, dataset_name)
        return

    y_true = np.asarray([gt_by_pair[key] for key in ordered_keys])
    scores = np.asarray([scores_by_pair[key] for key in ordered_keys])
    logger.info("Loaded %d ground truth and scores for %s (dataset=%s).", len(y_true), method_name, dataset_name)

    # Skip if Mongo already has matching record (dataset, method, n_samples).
    existing = store.get_record(dataset_name, method_name)
    if existing and int(existing.get("n_samples", -1)) == len(ordered_keys):
        logger.info(
            "Skipping %s (dataset=%s) - cached PAN metrics for n_samples=%d.",
            method_name,
            dataset_name,
            len(ordered_keys),
        )
        return

    # guarantees identical repeated CV splits for any methods that share the same sample set and ordering.
    # Ordering is set above given that method have the same number of samples.
    # Hence, repeated CV scores are based on the same random split given the same number of samples.
    cv_result = evaluator.compute_cv(
        y_true=y_true,
        scores=scores,
        ci_level=CI_LEVEL,
        n_boot=N_BOOT,
    )
    logger.info("Computed PAN metrics for %s (dataset=%s): %s", method_name, dataset_name, cv_result.metrics_mean)

    record = PANMetricsRecord(
        dataset_name=dataset_name,
        method_name=method_name,
        n_samples=int(len(ordered_keys)),
        metrics_mean=cv_result.metrics_mean,
        metrics_std=cv_result.metrics_std,
        metrics_ci=cv_result.metrics_ci,
        pan_metrics_per_split=[asdict(result) for result in cv_result.per_split],
        split_config=cv_result.split_config,
    )
    store.save_record(record)
    logger.info("Saved PAN metrics for %s (%d samples).", method_name, len(ordered_keys))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = PANDataLoader()
    split_manager = SplitManager(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    evaluator = PANEvaluator(PANMetricComputer(), split_manager)
    store = PANMetricsStore()

    for dataset_name in [CONFIG.STUDENT_ESSAYS, CONFIG.BLOG]:
        # normal methods with different impostor techniques
        for method in IMPOSTOR_METHODS:
            _compute_for_method(
                method,
                dataset_name=dataset_name,
                loader=loader,
                evaluator=evaluator,
                store=store,
                impostor_technique=None,
                n_impostors=N_IMPOSTORS,
                n_potential_impostors=N_POTENTIAL_IMPOSTORS,
                rounds=ROUNDS,
            )

        # ablation methods
        for method in ABLATION_METHODS:
            _compute_for_method(
                method,
                dataset_name=dataset_name,
                loader=loader,
                evaluator=evaluator,
                store=store,
                impostor_technique=IMPOSTOR_TECHNIQUE_FOR_ABLATIONS,
                n_impostors=N_IMPOSTORS,
                n_potential_impostors=N_POTENTIAL_IMPOSTORS,
                rounds=ROUNDS,
            )
