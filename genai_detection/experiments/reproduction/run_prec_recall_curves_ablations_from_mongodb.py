import argparse
import logging
import os
from typing import Dict

import pandas as pd

from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.experiments.run_prec_recall_curves_ablations import (
    LOCAL_SAVE_PATH,
)
from genai_detection.experiments.reproduction.ablation_args import ABLATION_DETECTORS
from genai_detection.experiments.reproduction.impostor_metrics import compute_metrics_for_thresholds
from genai_detection.experiments.reproduction.pan_metrics import PANDataLoader
from genai_detection.experiments.reproduction.prec_recall_curves import (
    compute_prec_recall_f1_acc_dict_on_existing_impostor_scores,
    plot_precision_recall_curve,
)
import logging

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def compute_prec_recall_curves_ablations_from_mongodb(
    dataset_name: str,
    impostor_technique: str = "in_domain",
    *,
    include_traditional: bool = True,
    batch_size: int = 250,
) -> Dict[str, object]:
    """
    Compute ablation precision–recall curves entirely from MongoDB-stored scores.
    """
    if impostor_technique not in IMPOSTOR_GENERATORS:
        raise ValueError(f"Unsupported impostor generator: {impostor_technique}")

    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    loader = PANDataLoader(mongo=mongoDB, batch_size=batch_size)

    metrics: Dict[str, pd.DataFrame] = {}
    n_pairs_per_method: Dict[str, int] = {}

    for variant in ABLATION_DETECTORS:
        scores_by_pair = loader.load_scores(
            method_name=variant,
            dataset_name=dataset_name,
            impostor_technique=impostor_technique,
        )
        n_pairs_per_method[variant] = len(scores_by_pair)
        if not scores_by_pair:
            logger.warning("No scores found for ablation '%s'.", variant)
            continue

        ordered_pairs = list(scores_by_pair.keys())
        gt_by_pair = loader.load_ground_truth(ordered_pairs, dataset_name)
        ordered_pairs = [pair for pair in ordered_pairs if pair in gt_by_pair]
        if not ordered_pairs:
            logger.warning(
                "No ground-truth found for ablation '%s' (dataset=%s).",
                variant,
                dataset_name,
            )
            continue
        if len(ordered_pairs) != len(scores_by_pair):
            logger.warning(
                "Missing ground-truth for %d/%d pairs for ablation '%s' (dataset=%s).",
                len(scores_by_pair) - len(ordered_pairs),
                len(scores_by_pair),
                variant,
                dataset_name,
            )

        ground_truth = [gt_by_pair[pair] for pair in ordered_pairs]
        scores = [scores_by_pair[pair] for pair in ordered_pairs]
        metrics[variant] = compute_metrics_for_thresholds(
            ground_truth=ground_truth,
            scores=scores,
            thresholds=CONFIG.THRESHOLDS,
        )
        logger.info(
            "Results for ablation %s (pairs=%d): %s",
            variant,
            len(scores),
            metrics[variant],
        )

    if n_pairs_per_method:
        logger.info("Pairs per ablation (dataset=%s): %s", dataset_name, n_pairs_per_method)

    if include_traditional:
        imp_generation_techniques = [impostor_technique, "two_step_llm"]
        traditional_results = compute_prec_recall_f1_acc_dict_on_existing_impostor_scores(
            dataset_name=dataset_name,
            imp_gen_techniques=imp_generation_techniques,
            include_baselines=False,
            save_artifacts=False,
        )
        for technique in imp_generation_techniques:
            if technique in traditional_results:
                metrics[technique] = traditional_results[technique]
            else:
                logger.warning(
                    "No precomputed %s results found for dataset %s.",
                    technique,
                    dataset_name,
                )

    return {
        "metrics": metrics,
    }


if __name__ == "__main__":
    label_translations = dict(CONFIG.LABEL_TRANSLATIONS)
    label_translations["in_domain"] = "Koppel and Winter, 2014"
    label_translations["two_step_llm"] = "LLM-based impostors"

    for dataset_name in [CONFIG.BLOG, CONFIG.STUDENT_ESSAYS]:
        results = compute_prec_recall_curves_ablations_from_mongodb(
            dataset_name=dataset_name,
            impostor_technique="in_domain",
            include_traditional=True,
        )
        results_dict = results.get("metrics", {})
        if not results_dict:
            logger.warning("No metrics computed for dataset %s.", dataset_name)
            continue

        logger.info("Results for dataset %s as dictionary:", dataset_name)
        logger.info(results_dict)

        metrics_df = pd.concat(results_dict, names=["method"]).reset_index(level=0)
        metrics_df.to_csv(
            LOCAL_SAVE_PATH / f"effectiveness_scores_{dataset_name}.csv",
            index=False,
        )
        logger.info(
            "Saved effectiveness scores to %s",
            LOCAL_SAVE_PATH / f"effectiveness_scores_{dataset_name}.csv",
        )

        plot_precision_recall_curve(
            results=results_dict,
            dataset_name=dataset_name,
            save_path=LOCAL_SAVE_PATH,
            title="Precision–Recall Curve",
            label_translations=label_translations,
            filename_extra="ablations"
        )
