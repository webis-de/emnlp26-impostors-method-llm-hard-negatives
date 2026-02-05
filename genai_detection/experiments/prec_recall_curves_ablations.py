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

from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.experiments.reproduction.impostor_metrics import (compute_metrics_parallel, load_all_pairs, )

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ablations import (ASGALFImpostorDetector, Potha2017ImpostorDetector,
                       )

logger = logging.getLogger(__name__)


ABLATION_DETECTORS = {
    "asgalf": ASGALFImpostorDetector,
    # "potha2017": Potha2017ImpostorDetector,
    # "homotopy": HBCImpostorDetector,
}

ABLATION_ARGS = {
        "asgalf":{
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
    Path(__file__).resolve().parents[3]
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
        detector = detector_cls(
            impostor_technique=impostor_technique,
            # n_impostors=n_impostors,
            **ABLATION_ARGS[name],
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
    metics_df = pd.DataFrame(compute_prec_recall_curves_ablations(dataset_name=CONFIG.BLOG))
    logger.info(metics_df)

    metics_df.to_csv(LOCAL_SAVE_PATH / "effectiveness_scores.csv", index=False)
    logger.info(f"Saved effectiveness scores as csv to {LOCAL_SAVE_PATH}/effectiveness_scores.csv.")
