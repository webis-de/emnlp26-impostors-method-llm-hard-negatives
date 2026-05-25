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

"""Compute and store PAN metrics for selected methods."""

from __future__ import annotations

import logging

import numpy as np

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pan_metrics.pan_cv import (
    PANEvaluator,
    SplitManager,
    compute_aligned_pair_keys_hash,
)
from genai_detection.experiments.reproduction.pan_metrics.pan_data_loader import PANDataLoader
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import PANMetricComputer
from genai_detection.experiments.reproduction.pan_metrics.pan_storage import PANMetricsStore

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

BASELINE_METHODS = [
    "unsupervised_baseline_min-max",
    "unsupervised_baseline_cosine",
    "supervised_baseline",
    "unmasking",
    "ppmd",
]

N_IMPOSTORS = 50
N_POTENTIAL_IMPOSTORS = None
ROUNDS = 100

# Bouckaert & Frank (2004): 10x 10-fold CV is best in terms of type I and II error as well as replicability..
N_SPLITS = 10
N_REPEATS = 10
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
    n_samples = len(ordered_keys)
    aligned_pair_keys_hash = compute_aligned_pair_keys_hash(ordered_keys)
    logger.info("Loaded %d ground truth and scores for %s (dataset=%s).", n_samples, method_name, dataset_name)

    _ = evaluator.get_or_compute_record(
        dataset_name=dataset_name,
        method_name=method_name,
        y_true=y_true,
        scores=scores,
        n_samples=n_samples,
        n_splits=split_manager.n_splits,
        n_repeats=split_manager.n_repeats,
        ci_level=CI_LEVEL,
        n_boot=N_BOOT,
        aligned_pair_keys_hash=aligned_pair_keys_hash,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = PANDataLoader()
    split_manager = SplitManager(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    store = PANMetricsStore()
    evaluator = PANEvaluator(metric_computer=PANMetricComputer(), split_manager=split_manager, store=store)

    for dataset_name in [CONFIG.STUDENT_ESSAYS, CONFIG.BLOG]:
        input_args = {
                "dataset_name":dataset_name,
                "loader":loader,
                "evaluator":evaluator,
                "store":store,
                "impostor_technique":None,
                "n_impostors":N_IMPOSTORS,
                "n_potential_impostors":N_POTENTIAL_IMPOSTORS,
                "rounds":ROUNDS,
                }

        for method in BASELINE_METHODS + IMPOSTOR_METHODS + ABLATION_METHODS:
            logger.info("Computing PAN metrics for %s method (dataset=%s).", method, dataset_name)
            # ablation methods
            if method in ABLATION_METHODS:
                input_args["impostor_technique"] = IMPOSTOR_TECHNIQUE_FOR_ABLATIONS
            else:
                # normal methods with different impostor techniques or baselines
                input_args["impostor_technique"] = None

            _compute_for_method(
                method,
                **input_args,
            )
