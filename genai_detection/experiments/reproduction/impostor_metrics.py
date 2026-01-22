import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Iterable, Tuple

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

N_PAIRS = 50 # TODO: increase

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def _load_input_pairs(dataset_name: str, split:str="all", balanced:bool=True) -> Tuple[List[str], List[int]]:
    """
    Load text-pair IDs and ground-truth labels from MongoDB.
    """
    logger.info(f"Loading input pairs for dataset '{dataset_name}'")
    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    collection = mongoDB.all_pairs_collection if split == "all" else mongoDB.test_pairs_collection if split == "test" else mongoDB.train_pairs_collection
    diff_pairs = list(
        mongoDB.find_document_by_multiple_fields(
            collection=collection,
            search_args=
            {"dataset_name": dataset_name,
             "same":False},
        )
    )#[:N_PAIRS//2]
    same_pairs =list(
        mongoDB.find_document_by_multiple_fields(
            collection=collection,
            search_args=
            {"dataset_name": dataset_name,
             "same":True},
        )
    )#[:N_PAIRS//2]

    if balanced:
        min_len = min(len(same_pairs), len(diff_pairs))
        test_pairs = diff_pairs[:min_len] + same_pairs[:min_len]
    else:
        test_pairs = diff_pairs + same_pairs

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

def load_test_pairs(dataset_name: str) -> Tuple[List[str], List[int]]:
    """
    Load text-pair IDs and ground-truth labels from MongoDB.
    """
    return _load_input_pairs(dataset_name, split="test")

def load_train_pairs(dataset_name: str) -> Tuple[List[str], List[int]]:
    return _load_input_pairs(dataset_name, split="train")

def load_all_pairs(dataset_name: str) -> Tuple[List[str], List[int]]:
    return _load_input_pairs(dataset_name, split="all")

# ---------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------

def compute_metrics_for_binary_predictions(
        ground_truth: List[int],
        binary_preds: List[int],
        threshold: float
):
    return {
                "threshold": threshold,
                "precision": precision_score(
                    ground_truth,
                    binary_preds,
                    labels=[0, 1],
                    average=None,
                    zero_division=0,
                ),
                "recall": recall_score(
                    ground_truth,
                    binary_preds,
                    labels=[0, 1],
                    average=None,
                    zero_division=0,
                ),
                "f1": f1_score(
                    ground_truth,
                    binary_preds,
                    labels=[0, 1],
                    average=None,
                    zero_division=0,
                ),
                "accuracy": accuracy_score(
                    ground_truth,
                    binary_preds,
                ),
            }

def compute_metrics_for_thresholds(
    ground_truth: List[int],
    scores: List[float],
    thresholds: Iterable[float] = CONFIG.THRESHOLDS,
) -> pd.DataFrame:
    """
    Compute precision, recall, f1, accuracy for a list of thresholds.
    """
    rows = []
    assert len(ground_truth) == len(scores), f"Length of ground_truth and scores do not match: gt {len(ground_truth)}, scores {len(scores)}"

    for t in thresholds:
        binary_preds = [1 if s >= t else 0 for s in scores]
        rows.append(
            compute_metrics_for_binary_predictions(
                ground_truth=ground_truth, binary_preds=binary_preds, threshold=t)
        )

    return pd.DataFrame(rows)


def compute_metrics_parallel(
    ground_truth: List[int],
    predictions: Dict[str, List[float]],
    thresholds: Iterable[float] = CONFIG.THRESHOLDS,
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

    def safe_get(arr):
        if len(arr) > positive_class_id:
            return arr[positive_class_id]
        return 0.0

    return float(df[metric].apply(safe_get).max())

def extract_best_metric_and_position(
        df: pd.DataFrame,
        metric: str,
        positive_class_id: int = 1,
) -> dict:
    """
    Extract the maximum metric value for the given class across thresholds
    and return its position in the DataFrame.
    """
    if metric == "accuracy":
        best_idx = df["accuracy"].idxmax()
        return {
            "best_score": float(df.loc[best_idx, "accuracy"]),
            "best_idx": int(best_idx)
        }

    def safe_get(arr):
        if len(arr) > positive_class_id:
            return float(arr[positive_class_id])
        return 0.0

    # build scalar series for the selected class
    class_metric_values = df[metric].apply(safe_get)

    best_idx = class_metric_values.idxmax()

    return {
        "best_score": float(class_metric_values.loc[best_idx]),
        "best_idx": int(best_idx)
    }
