"""Compute pairwise PAN metric significance on aligned intersections."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pan_metrics.pan_cv import PANEvaluator, SplitManager
from genai_detection.experiments.reproduction.pan_metrics.pan_data_loader import PANDataLoader
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import PANMetricComputer
from genai_detection.experiments.reproduction.pan_metrics.pan_significance import PANPairwiseSignificance
from genai_detection.experiments.reproduction.pan_metrics.pan_storage import PANMetricsStore

logger = logging.getLogger(__name__)


IMPOSTOR_TECHNIQUE_FOR_ABLATIONS = "in_domain"

# Comment out methods you don't want to run.
METHODS = [
    "two_step_llm",
    "on_the_fly_chatnoir",
    # "on_the_fly_serpapi",
    "on_the_fly_startpage",
    "in_domain",
    # "one_step_llm",
    # "translation",
    # "mirror_minds",   # raises error
    "bdi",
    "homotopy",
    "potha2017",
    "asgalf",
    "std_impostor",
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


def _save_results(results: dict, save_dir: Path, dataset_name: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"pan_metrics_significance_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return out_path


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

    significance = PANPairwiseSignificance(
        data_loader=loader,
        evaluator=evaluator,
        store=store,
        split_manager=split_manager,
    )

    for dataset_name in [CONFIG.STUDENT_ESSAYS, CONFIG.BLOG]:


        results = significance.compute_pairwise_significance(
            dataset_name=dataset_name,
            methods=METHODS,
            impostor_technique=IMPOSTOR_TECHNIQUE_FOR_ABLATIONS,
            n_impostors=N_IMPOSTORS,
            n_potential_impostors=N_POTENTIAL_IMPOSTORS,
            rounds=ROUNDS,
            ci_level=CI_LEVEL,
            n_boot=N_BOOT,
        )

        out_path = _save_results(results, store.save_dir / "statistical_significance", dataset_name=dataset_name)
        logger.info("Saved significance results to %s", out_path)
