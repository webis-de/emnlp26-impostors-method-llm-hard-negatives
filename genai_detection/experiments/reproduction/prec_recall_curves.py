# Copyright 2025 Klara M. Gutekunst, Webis
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

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from bson import ObjectId
from matplotlib import pyplot as plt
from sklearn.metrics import auc

from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.detectors.impostor_supervised_baseline import SupervisedImpostorBaseline
from genai_detection.detectors.impostor_unsupervised_baseline import UnSupervisedImpostorBaseline
from genai_detection.detectors.ppmd import PPMdDetector
from genai_detection.detectors.unmasking import UnmaskingDetector
from genai_detection.experiments.reproduction.impostor_metrics import (compute_metrics_for_thresholds,
                                                                       compute_metrics_parallel, load_all_pairs, )
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[3]
    / CONFIG.SAVE_PATH
    / "reproduction"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def _build_baselines(dataset_name: str):
    return {
        "unsupervised_baseline_min-max": UnSupervisedImpostorBaseline(
            use_cosine_simiarity=False,
            dataset_name=dataset_name,
        ),
        "unsupervised_baseline_cosine": UnSupervisedImpostorBaseline(
            use_cosine_simiarity=True,
            dataset_name=dataset_name,
        ),
        "supervised_baseline": SupervisedImpostorBaseline(
            dataset_name=dataset_name
        ),
        "unmasking": UnmaskingDetector(),
        "ppmd": PPMdDetector(),
    }

def _iter_batches(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def _score_baselines_for_pair_batches(
    baselines: Dict[str, object],
    pair_batches: Iterable[List[Tuple[str, str]]],
) -> Dict[str, List[float]]:
    predictions: Dict[str, List[float]] = {name: [] for name in baselines.keys()}

    for batch in pair_batches:
        if not batch:
            continue
        flat_texts = [text for pair in batch for text in pair]
        for name, baseline in baselines.items():
            preds = baseline.get_score(flat_texts)
            preds = preds.tolist() if hasattr(preds, "tolist") else preds
            predictions[name].extend(np.asarray(preds).ravel().tolist())

    return predictions

def _compute_metrics_for_predictions(
    ground_truth: List[int],
    predictions: Dict[str, List[float]],
) -> Dict[str, pd.DataFrame]:
    logger.info(
        "Computing metrics for %d approaches",
        len(predictions),
    )
    return compute_metrics_parallel(
        ground_truth,
        predictions,
    )

def compute_prec_recall_f1_acc_dict(
    dataset_name: str,
    imp_gen_techniques: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Reproduction of Figures 4a and 4b from Koppel et al. (2014).
    """
    PAIR_BATCH_SIZE = 256  # number of PAIRS per batch (must be int)
    TEXT_BATCH_SIZE = PAIR_BATCH_SIZE * 2

    logger.info(
        "Reproducing Figure 4 with techniques: %s",
        imp_gen_techniques,
    )

    assert all(
        t in IMPOSTOR_GENERATORS for t in imp_gen_techniques
    ), f"Unsupported impostor generator in {imp_gen_techniques}"

    # -----------------------------------------------------------------
    # Load test data
    # -----------------------------------------------------------------

    text_id_pairs, ground_truth = load_all_pairs(dataset_name)
    text_id_pairs_len = len(text_id_pairs)
    assert text_id_pairs_len % 2 == 0, "Flattened text list must contain even number of elements but length is {}".format(text_id_pairs_len)
    logger.info("Number of texts used %d, number of pairs %d",text_id_pairs_len, text_id_pairs_len//2)

    # -----------------------------------------------------------------
    # Collect predictions (batched over texts, but compute all techniques per batch)
    # -----------------------------------------------------------------

    predictions: Dict[str, List[float]] = {tech: [] for tech in imp_gen_techniques}
    # Build detectors once (avoid re-instantiating each batch)
    detectors = {
        technique: ImpostorDetector(
            impostor_technique=technique,
            n_impostors=50,
            dataset_name=dataset_name,
        )
        for technique in imp_gen_techniques
    }

    for start in range(0, text_id_pairs_len, TEXT_BATCH_SIZE):
        batch_texts = text_id_pairs[start: start + TEXT_BATCH_SIZE]

        # Ensure we never split a pair (should be guaranteed by TEXT_BATCH_SIZE, but keep safe)
        if len(batch_texts) % 2 != 0:
            raise ValueError(
                f"Batch starting at {start} has odd number of texts ({len(batch_texts)}), would break pairing."
            )

        batch_pair_count = len(batch_texts) // 2
        logger.info(
            "Scoring batch: texts [%d:%d] -> %d pairs",
            start,
            min(start + TEXT_BATCH_SIZE, text_id_pairs_len),
            batch_pair_count,
        )

        for technique in imp_gen_techniques:
            detector = detectors[technique]
            batch_scores = detector.get_score(text=batch_texts)
            assert batch_scores is not None
            if len(batch_scores) != batch_pair_count:
                raise ValueError(
                    f"Technique {technique}: expected {batch_pair_count} scores, got {len(batch_scores)}"
                )

            predictions[technique].extend(batch_scores)

    # Sanity check: all techniques produced full-length results
    total_pairs = text_id_pairs_len // 2
    for technique in imp_gen_techniques:
        if len(predictions[technique]) != total_pairs:
            raise ValueError(
                f"Technique {technique}: expected {total_pairs} total scores, got {len(predictions[technique])}"
            )

    baselines = _build_baselines(dataset_name=dataset_name)

    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    text_test_pairs = mongoDB.get_texts_for_ids(text_ids=text_id_pairs)

    text_pairs = list(zip(text_test_pairs[0::2], text_test_pairs[1::2]))
    baseline_predictions = _score_baselines_for_pair_batches(
        baselines=baselines,
        pair_batches=[text_pairs],
    )
    predictions.update(baseline_predictions)
    # -----------------------------------------------------------------
    # Metric computation (shared implementation)
    # -----------------------------------------------------------------

    results = _compute_metrics_for_predictions(
        ground_truth=ground_truth,
        predictions=predictions,
    )

    return results

def compute_prec_recall_f1_acc_dict_on_existing_impostor_scores(
    dataset_name: str,
    imp_gen_techniques: List[str],
    *,
    n_impostors: int = 50,
    n_potential_impostors: Optional[int] = None,
    rounds: int = 100,
    batch_size: int = 250,
) -> Dict[str, pd.DataFrame]:
    """
    Same as compute_prec_recall_f1_acc_dict, but loads precomputed
    impostor scores from the MongoDB impostor_outputs collection instead of active computation.
    Baselines are computed on the fly.
    """
    logger.info(
        "Reproducing Figure 4 from stored impostor outputs for %s",
        imp_gen_techniques,
    )

    assert all(
        t in IMPOSTOR_GENERATORS for t in imp_gen_techniques
    ), f"Unsupported impostor generator in {imp_gen_techniques}"
    if not imp_gen_techniques:
        raise ValueError("imp_gen_techniques must not be empty.")

    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

    results = {}

    def _check_id_type(value):
        assert isinstance(value, ObjectId), f"{value} is not an ObjectId, but of type {type(value)}"
        return value

    def _load_scores_for_technique(technique: str) -> Dict[Tuple[ObjectId, ObjectId], float]:
        query = {
            "impostor_generation_technique": technique,
            "n_impostors": n_impostors,
        }
        if n_potential_impostors is not None:
            query["n_potential_impostors"] = n_potential_impostors
        cursor = mongoDB.impostor_output_collection.find(
            query,
            {"left_id": 1, "right_id": 1, "scores_over_different_rounds": 1},
            batch_size=batch_size,
        ).sort("_id", 1)
        scores_by_pair: Dict[Tuple[ObjectId, ObjectId], float] = {}
        for doc in cursor:
            left_id = _check_id_type(doc["left_id"])
            right_id = _check_id_type(doc["right_id"])
            pair = (left_id, right_id)
            if pair not in scores_by_pair:
                scores_by_pair[pair] = doc["scores_over_different_rounds"] / rounds
        logger.info(
            "Loaded %d scores for technique %s",
            len(scores_by_pair),
            technique,
        )
        return scores_by_pair

    def _load_ground_truth_for_pairs(
        keys: List[Tuple[ObjectId, ObjectId]],
    ) -> Dict[Tuple[ObjectId, ObjectId], int]:
        gt_by_pair: Dict[Tuple[ObjectId, ObjectId], int] = {}
        for batch in _iter_batches(keys, batch_size):
            or_conditions = [{"left_id": l, "right_id": r} for l, r in batch]
            cursor = mongoDB.all_pairs_collection.find(
                {"dataset_name": dataset_name, "$or": or_conditions},
                {"left_id": 1, "right_id": 1, "same": 1},
                batch_size=batch_size,
            )
            for doc in cursor:
                gt_by_pair[(doc["left_id"], doc["right_id"])] = int(doc["same"])
        return gt_by_pair

    for technique in imp_gen_techniques:
        # key: (left_id, right_id)
        loaded_scores_by_technique = _load_scores_for_technique(technique)
        if len(loaded_scores_by_technique) > 0:
            loaded_keys = list(loaded_scores_by_technique.keys())
            gt_by_pair = _load_ground_truth_for_pairs(loaded_keys)
            ordered_keys = sorted(
                key for key in loaded_keys if key in gt_by_pair
            )
            if len(ordered_keys) != len(loaded_scores_by_technique):
                logger.warning(
                    "Missing ground-truth for %d pairs (technique=%s, dataset=%s); dropping them. Since the "
                    "'dataset_name' is not saved in the 'impostors_outputs' collection, this could lead to these "
                    "drops (not ID inconsistencies).",
                    len(loaded_scores_by_technique) - len(ordered_keys),
                    technique, dataset_name,
                )
            else:
                logger.info("All loaded %d pairs (technique=%s) have a ground truth.",
                    len(loaded_scores_by_technique) ,
                    technique,)
            ground_truth = [gt_by_pair[key] for key in ordered_keys]
            predictions = [loaded_scores_by_technique[key] for key in ordered_keys]

            results[technique] = compute_metrics_for_thresholds(
                ground_truth=ground_truth,
                scores=predictions,
                thresholds=CONFIG.THRESHOLDS,
            )
            logger.info(f"Results for {technique}: {results[technique]}")
        else:
            logger.warning(f"No scores for technique {technique} found in mongoDB.")

    baselines = _build_baselines(dataset_name=dataset_name)
    # load all pairs per dataset
    text_id_pairs, ground_truth = load_all_pairs(dataset_name)
    text_id_pairs_len = len(text_id_pairs)
    assert (
        text_id_pairs_len % 2 == 0
    ), "Flattened text list must contain even number of elements but length is {}".format(
        text_id_pairs_len
    )
    logger.info(
        "Number of texts used %d, number of pairs %d",
        text_id_pairs_len,
        text_id_pairs_len // 2,
    )

    text_test_pairs = mongoDB.get_texts_for_ids(text_ids=text_id_pairs)

    text_pairs = list(zip(text_test_pairs[0::2], text_test_pairs[1::2]))
    baseline_predictions = _score_baselines_for_pair_batches(
        baselines=baselines,
        pair_batches=[text_pairs],
    )

    for baseline, pred in baseline_predictions.items():
        results[baseline] = compute_metrics_for_thresholds(
            ground_truth=ground_truth,
            scores=baseline_predictions[baseline],
            thresholds=CONFIG.THRESHOLDS,
        )
        logger.info(f"Results for {baseline}: {results[baseline]}")

    return results

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_precision_recall_curve(
    results: Dict[str, pd.DataFrame],
    dataset_name: str,
    save_path: Path = None,
    title: str = None,
):
    """
    Precision–Recall curves (Figures 4a, 4b in Koppel et al., 2014).
    """
    for positive_class_id in [0, 1]:
        fig = plt.figure(figsize=(10, 7))
        positive_class = (
            "Same Author" if positive_class_id == 1 else "Different Author"
        )

        for key, df in results.items():
            label = CONFIG.LABEL_TRANSLATIONS.get(key, key)
            color = CONFIG.LABEL_COLORS.get(key, "black")
            if "precision" in df.columns and "recall" in df.columns:
                precisions = df["precision"].apply(
                    lambda x: x[positive_class_id]
                )
                recalls = df["recall"].apply(
                    lambda x: x[positive_class_id]
                )

                # filter out points where both precision and recall are zero
                mask = ~(
                    ((precisions == 0) & (recalls == 0))
                    | ((precisions == 1) & (recalls == 0))
                    | ((precisions == 0) & (recalls == 1))
                )

                precisions = precisions[mask]
                recalls = recalls[mask]
                unique_pairs = set(zip(recalls, precisions))

                logger.info("%s: class %d", key, positive_class_id)
                logger.info("Number of unique (x, y) pairs: %d", len(unique_pairs))

                plt.plot(
                    recalls,
                    precisions,
                    # marker="o",
                    # markersize=4,
                    color=color,
                    label=label,
                )

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim(0, 1)
        plt.gca().set_aspect("equal")
        plt.xlabel("Recall $\\frac{TP}{TP + FN}$", fontsize=14)
        plt.ylabel("Precision $\\frac{TP}{TP + FP}$", fontsize=14)
        if title is None:
            title = "Precision–Recall Curve Across Impostor Generation Techniques"
        title += f"\nDataset: {CONFIG.DATASET_TRANSLATIONS[dataset_name]} ({positive_class})"    # subtitle
        plt.title(title)
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.legend()

        for format in ["pdf", "svg"]:
            fname = (
                f"roc_prec_recall_curve_{dataset_name.replace(' ', '_')}_"
                f"{positive_class.lower().replace(' ', '_')}.{format}"
            )
            if not save_path:
                logger.warning(f"No save path for {fname}, using default save path: {LOCAL_SAVE_PATH}")
                save_path = LOCAL_SAVE_PATH
            plt.savefig(
                save_path / fname,
                bbox_inches="tight",
            )
            logger.info("Saved plot to %s", save_path/fname)

        fig.clear()


def _extract_best_pr_points_per_impostor(
    results: Dict[str, pd.DataFrame],
    dataset_name: str,
) -> pd.DataFrame:
    rows = []

    for impostor_method, df in results.items():

        for class_id, class_name in [(0, "Different Author"), (1, "Same Author")]:
            if "precision" in df.columns and "recall" in df.columns:
                precisions = df["precision"].apply(lambda x: x[class_id]).to_numpy()
                recalls = df["recall"].apply(lambda x: x[class_id]).to_numpy()
                thresholds = df["threshold"].to_numpy()

                # Sort by recall for valid PR AUC
                order = np.argsort(recalls)
                recalls_sorted = recalls[order]
                precisions_sorted = precisions[order]

                pr_auc = auc(recalls_sorted, precisions_sorted)

                # ---- Best precision (tie → recall) ----
                best_p_idx = np.lexsort((-recalls, -precisions))[0]

                # ---- Best recall (tie → precision) ----
                best_r_idx = np.lexsort((-precisions, -recalls))[0]

                # ---- Best PR operating point (proxy for PR-AUC) ----
                pr_product = precisions * recalls
                best_auc_idx = pr_product.argmax()

                rows.append({
                    "dataset_name": dataset_name.replace("_", " ").capitalize(),
                    "impostor_generation": CONFIG.LABEL_TRANSLATIONS[impostor_method],
                    "class": class_name,

                    "n_test_samples": 50,   # TODO: adjust

                    "best_precision": precisions[best_p_idx],
                    "best_precision_recall": recalls[best_p_idx],
                    "best_precision_threshold": thresholds[best_p_idx],

                    "best_recall": recalls[best_r_idx],
                    "best_recall_precision": precisions[best_r_idx],
                    "best_recall_threshold": thresholds[best_r_idx],

                    "best_auc_precision": precisions[best_auc_idx],
                    "best_auc_recall": recalls[best_auc_idx],
                    "best_auc_threshold": thresholds[best_auc_idx],

                    "pr_auc": pr_auc,
                })

    return pd.DataFrame(rows)

def run_prec_recall_curves(dataset_name:str, imp_gen_techniques:List[str], compute_score_fn=compute_prec_recall_f1_acc_dict):
    results_dict = compute_score_fn(dataset_name=dataset_name, imp_gen_techniques=imp_gen_techniques)
    logger.info("Obtained scores for approaches %s", results_dict.keys())

    # results_dict: {approach_name: DataFrame}
    dfs = []
    for approach, df in results_dict.items():
        temp_df = df.copy()
        temp_df["approach"] = approach
        dfs.append(temp_df)

    # Combine all approaches
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save to CSV
    combined_df.to_csv(LOCAL_SAVE_PATH / "effectiveness_scores.csv", index=False)
    logger.info(f"Saved effectiveness scores as csv to {LOCAL_SAVE_PATH}/effectiveness_scores.csv.")

    # best PR operating points
    best_pr_df = _extract_best_pr_points_per_impostor(
        results=results_dict,
        dataset_name=dataset_name,
    )

    best_pr_df.to_csv(
        LOCAL_SAVE_PATH / f"best_precision_recall_points_{dataset_name}.csv",
        index=False,
        float_format="%.2f"
    )
    logger.info(
        "Saved best precision–recall operating points to %s",
        LOCAL_SAVE_PATH / "best_precision_recall_points.csv",
    )

    plot_precision_recall_curve(results=results_dict, dataset_name=dataset_name)
