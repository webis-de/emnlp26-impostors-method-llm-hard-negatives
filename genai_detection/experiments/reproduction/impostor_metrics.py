import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

THRESHOLDS = np.arange(0.0, 1.05, 0.1)

LABEL_TRANSLATIONS = {
    "in_domain": "In-Domain",
    "on_the_fly": "On-the-Fly",
    "one_step_llm": "One-Step Paraphraser (LLM)",
    "two_step_llm": "Two-Step Paraphraser (LLM)",
    "unsupervised_baseline_min-max": "Unsup. Min-Max (B)",
    "unsupervised_baseline_cosine": "Unsup. Cosine (B)",
    "supervised_baseline": "Sup. SVM (B)",
    "unmasking": "Unmasking",
    "ppmd": "PPMd",
    "translation": "Translation",
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_test_pairs(dataset_name: str) -> Tuple[List[str], List[int]]:
    """
    Load text-pair IDs and ground-truth labels from MongoDB.
    """
    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

    test_pairs = list(
        mongoDB.find_document_by_non_id_field(
            collection=mongoDB.test_pairs_collection,
            document_field_name="dataset_name",
            document_value=dataset_name,
        )
    )

    text_ids = [
        str(id_)
        for pair in test_pairs
        for id_ in (pair["left_id"], pair["right_id"])
    ]
    ground_truth = [pair["same"] for pair in test_pairs]

    logger.info(
        "Loaded %d test pairs for dataset '%s'",
        len(ground_truth),
        dataset_name,
    )

    return text_ids, ground_truth


# ---------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------

def compute_metrics_for_thresholds(
    ground_truth: List[int],
    scores: List[float],
    thresholds: Iterable[float] = THRESHOLDS,
) -> pd.DataFrame:
    """
    Compute precision, recall, f1, accuracy for a list of thresholds.
    """
    rows = []

    for t in thresholds:
        binary_preds = [1 if s >= t else 0 for s in scores]

        rows.append(
            {
                "threshold": t,
                "precision": precision_score(
                    ground_truth,
                    binary_preds,
                    average=None,
                    zero_division=0,
                ),
                "recall": recall_score(
                    ground_truth,
                    binary_preds,
                    average=None,
                    zero_division=0,
                ),
                "f1": f1_score(
                    ground_truth,
                    binary_preds,
                    average=None,
                    zero_division=0,
                ),
                "accuracy": accuracy_score(
                    ground_truth,
                    binary_preds,
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_metrics_parallel(
    ground_truth: List[int],
    predictions: Dict[str, List[float]],
    thresholds: Iterable[float] = THRESHOLDS,
) -> Dict[str, pd.DataFrame]:
    """
    Compute metrics for multiple approaches in parallel.
    """

    def _worker(args):
        name, scores = args
        logger.info(f"Compute metrics for {name}. Input scores are None: {scores is None}. Thresholds are None: {thresholds is None}")
        return name, compute_metrics_for_thresholds(
            ground_truth, scores, thresholds
        )

    results = {}
    with ThreadPoolExecutor() as executor:
        for name, df in executor.map(_worker, predictions.items()):
            results[name] = df
            logger.info(f"Computed metrics for {name}. Obtained metrics are None: {results[name] is None}")

    return results


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def extract_best_metric(
    df: pd.DataFrame,
    metric: str,
    positive_class_id: int = 1,
) -> float:
    """
    Extract the maximum metric value across thresholds.
    """
    if metric == "accuracy":
        return float(df["accuracy"].max())

    return float(
        df[metric]
        .apply(lambda arr: arr[positive_class_id])
        .max()
    )
