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
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc

from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.detectors.impostor_supervised_baseline import SupervisedImpostorBaseline
from genai_detection.detectors.impostor_unsupervised_baseline import UnSupervisedImpostorBaseline
from genai_detection.detectors.ppmd import PPMdDetector
from genai_detection.detectors.unmasking import UnmaskingDetector
from genai_detection.experiments.reproduction.impostor_metrics import (
    compute_metrics_parallel,
    load_test_pairs, load_all_pairs,
)
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

def compute_prec_recall_f1_acc_dict(
    dataset_name: str,
    imp_gen_techniques: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Reproduction of Figures 4a and 4b from Koppel et al. (2014).
    """

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

    text_test_ID_pairs, ground_truth = load_all_pairs(dataset_name)
    logger.info("Number of texts used %d, number of pairs %d",len(text_test_ID_pairs), len(text_test_ID_pairs)//2)

    # -----------------------------------------------------------------
    # Collect predictions
    # -----------------------------------------------------------------

    predictions: Dict[str, List[float]] = {}

    for technique in imp_gen_techniques:
        logger.info("Obtaining impostor scores: %s", technique)
        if technique == "on_the_fly":
            # filter such that 50 same and different author pairs are tested; based on ground truth Boolean values
            sampled_test_ID_pairs, ground_truth = load_test_pairs(dataset_name)
        detector = ImpostorDetector(
            impostor_technique=technique,
            n_impostors=50,
            dataset_name=dataset_name,
        )
        scores = detector.get_score(text=text_test_ID_pairs)
        assert scores is not None
        predictions[technique] = scores

    baselines = {
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

    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    text_test_pairs = mongoDB.get_texts_for_ids(text_ids=text_test_ID_pairs)

    for name, baseline in baselines.items():
        # baselines assume text is raw text, not text ID
        print(name)

        preds = baseline.get_score(text_test_pairs)
        preds = preds.tolist() if hasattr(preds, "tolist") else preds
        predictions[name] = np.asarray(preds).ravel().tolist()
        print(predictions[name])
    # -----------------------------------------------------------------
    # Metric computation (shared implementation)
    # -----------------------------------------------------------------

    logger.info(
        "Computing metrics for %d approaches",
        len(predictions),
    )

    results = compute_metrics_parallel(
        ground_truth,
        predictions,
    )

    return results


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_precision_recall_curve(
    results: Dict[str, pd.DataFrame],
    dataset_name: str,
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
        plt.title(
            "Precision–Recall Curve Across Impostor Generation Techniques\n"
            f"Dataset: {dataset_name.capitalize()} ({positive_class})"
        )
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.legend()

        for format in ["pdf", "svg"]:
            fname = (
                f"roc_prec_recall_curve_{dataset_name.replace(' ', '_')}_"
                f"{positive_class.lower().replace(' ', '_')}.{format}"
            )
            plt.savefig(
                LOCAL_SAVE_PATH / fname,
                bbox_inches="tight",
            )
            logger.info("Saved %s to %s", fname, LOCAL_SAVE_PATH)

        fig.clear()



def _extract_best_pr_points_per_impostor(
    results: Dict[str, pd.DataFrame],
    dataset_name: str,
) -> pd.DataFrame:
    rows = []

    for impostor_method, df in results.items():

        for class_id, class_name in [(0, "Different Author"), (1, "Same Author")]:
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
                "dataset": dataset_name.replace("_", " ").capitalize(),
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

def run_prec_recall_curves(dataset_name:str, imp_gen_techniques:List[str]):
    results_dict = compute_prec_recall_f1_acc_dict(dataset_name=dataset_name, imp_gen_techniques=imp_gen_techniques)
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
