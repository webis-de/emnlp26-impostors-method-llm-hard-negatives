"""
Precision–recall evaluation for impostor ablations and the original method.

This experiment computes PR curves for all impostor ablations plus the
original impostor approach, while storing ablation scores in a dedicated
MongoDB collection.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ablations.std_impostor import StdImpostor
from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.experiments.reproduction.impostor_metrics import (compute_metrics_parallel, load_all_pairs, )
from genai_detection.experiments.reproduction.prec_recall_curves import plot_precision_recall_curve

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ablations import (
    ASGALFImpostorDetector,
    BDIImpostorDetector,
    HBCImpostorDetector,
    Potha2017ImpostorDetector,
)

logger = logging.getLogger(__name__)


ABLATION_DETECTORS = {
    "bdi": BDIImpostorDetector,
    "homotopy": HBCImpostorDetector,
    "potha2017": Potha2017ImpostorDetector,
    "asgalf": ASGALFImpostorDetector,
    "std_impostor": StdImpostor,
}

ABLATION_ARGS = {
        "bdi": {
            "rounds": 50,
            "portion_delete": 0.67,
            "n_impostors": 50,
        },
        "homotopy":{
            "rounds": 50,
            "n_impostors": 50,
            },
        "asgalf":{
            "rounds": 50,
            "portion_delete": 0.6,
            "n_impostors": 20,
            },
        "std_impostor": {
            "rounds": 50,
            "portion_delete": 0.6,
            "n_impostors": 20,
                },
        "potha2017": {
                "rounds": 10,
                "portion_delete": 0.5,
                "impostors_per_problem": 50,
                "impostors_per_round": 5,
                }
        }

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "ablations"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)


def _configure_ablation_detector(detector, variant: str) -> None:
    """
    Route detector outputs to the ablation collection and tag the variant.
    """
    detector.impostor_output_collection = (
        detector.mongoDB.impostor_ablation_output_collection
    )
    detector.impostor_variant = variant


def _build_ablation_detectors(
    dataset_name: str,
    impostor_technique: str,
    n_impostors: int,
) -> Dict[str, object]:
    """
    Create and configure detectors for all impostor ablations.
    """
    detectors: Dict[str, object] = {}
    for name, detector_cls in ABLATION_DETECTORS.items():
        args = ABLATION_ARGS[name]
        if ("n_impostors" not in args) and ("impostors_per_round" not in args):
            args[n_impostors] = n_impostors
        detector = detector_cls(
            impostor_technique=impostor_technique,
            **args,
            dataset_name=dataset_name,
        )
        _configure_ablation_detector(detector, variant=name)
        detectors[name] = detector
    return detectors


def _iter_text_batches(text_ids: List[str], pair_batch_size: int) -> Iterable[List[str]]:
    """
    Yield text-id batches while preserving pair boundaries.
    """
    batch_size = pair_batch_size * 2
    for start in range(0, len(text_ids), batch_size):
        batch = text_ids[start : start + batch_size]
        if len(batch) % 2 != 0:
            raise ValueError(
                f"Batch starting at {start} has odd number of texts ({len(batch)})."
            )
        yield batch


def _score_detectors(
    detectors: Dict[str, object],
    text_ids: List[str],
    pair_batch_size: int,
) -> Dict[str, List[float]]:
    """
    Compute scores for each detector over the full dataset.
    """
    predictions: Dict[str, List[float]] = {name: [] for name in detectors}

    for batch_texts in _iter_text_batches(text_ids, pair_batch_size):
        batch_pair_count = len(batch_texts) // 2
        for name, detector in detectors.items():
            batch_scores = detector.get_score(text=batch_texts)
            if batch_scores is None:
                raise ValueError(f"Detector {name} returned None scores.")
            if len(batch_scores) != batch_pair_count:
                raise ValueError(
                    f"{name}: expected {batch_pair_count} scores, got {len(batch_scores)}"
                )
            predictions[name].extend(np.asarray(batch_scores).ravel().tolist())

    return predictions


def compute_prec_recall_curves_ablations(
    dataset_name: str,
    impostor_technique: str = "in_domain",
    n_impostors: int = 50,
    pair_batch_size: int = 256,
) -> Dict[str, object]:
    """
    Compute precision–recall curves for all impostor ablations and the original method.
    """
    if impostor_technique not in IMPOSTOR_GENERATORS:
        raise ValueError(f"Unsupported impostor generator: {impostor_technique}")

    text_ids, ground_truth = load_all_pairs(dataset_name)
    detectors = _build_ablation_detectors(
        dataset_name=dataset_name,
        impostor_technique=impostor_technique,
        n_impostors=n_impostors,
    )
    predictions = _score_detectors(
        detectors=detectors,
        text_ids=text_ids,
        pair_batch_size=pair_batch_size,
    )

    return compute_metrics_parallel(
        ground_truth=ground_truth,
        predictions=predictions,
        thresholds=CONFIG.THRESHOLDS,
    )


__all__ = ["compute_prec_recall_curves_ablations"]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    for dataset_name in [CONFIG.BLOG, CONFIG.STUDENT_ESSAYS]:
        results_dict = compute_prec_recall_curves_ablations(dataset_name=dataset_name, impostor_technique="in_domain")
        logger.info("Results for dataset %s as dictionary:", dataset_name)
        logger.info(results_dict)
        # optional: flatten to a single DataFrame for CSV
        metrics_df = pd.concat(results_dict, names=["method"]).reset_index(level=0)
        logger.info("Dataframe of results:")
        logger.info(metrics_df)

        metrics_df.to_csv(LOCAL_SAVE_PATH / "effectiveness_scores.csv", index=False)
        logger.info(f"Saved effectiveness scores to {LOCAL_SAVE_PATH / 'effectiveness_scores.csv'}")

        plot_precision_recall_curve(results=results_dict, dataset_name=dataset_name, save_path=LOCAL_SAVE_PATH)
