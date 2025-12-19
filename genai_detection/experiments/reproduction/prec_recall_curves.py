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

from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.detectors.impostor_supervised_baseline import SupervisedImpostorBaseline
from genai_detection.detectors.impostor_unsupervised_baseline import UnSupervisedImpostorBaseline
from genai_detection.detectors.ppmd import PPMdDetector
from genai_detection.detectors.unmasking import UnmaskingDetector
from genai_detection.experiments.reproduction.impostor_metrics import (
    load_test_pairs,
    compute_metrics_parallel,
    LABEL_TRANSLATIONS,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

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

    text_test_pairs, ground_truth = load_test_pairs(dataset_name)

    # -----------------------------------------------------------------
    # Collect predictions
    # -----------------------------------------------------------------

    predictions: Dict[str, List[float]] = {}

    for technique in imp_gen_techniques:
        logger.info("Obtaining impostor scores: %s", technique)
        detector = ImpostorDetector(
            impostor_technique=technique,
            n_impostors=2,  # TODO: increase
        )
        scores = detector.get_score(text=text_test_pairs)
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

    for name, baseline in baselines.items():
        preds = baseline.get_score(text_test_pairs)
        preds = preds.tolist() if hasattr(preds, "tolist") else preds
        predictions[name] = np.asarray(preds).ravel().tolist()

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

    fig = plt.figure(figsize=(10, 7))

    for positive_class_id in [0, 1]:
        positive_class = (
            "Same Author" if positive_class_id == 1 else "Different Author"
        )

        for key, df in results.items():
            label = LABEL_TRANSLATIONS.get(key, key)
            precisions = df["precision"].apply(
                lambda x: x[positive_class_id]
            )
            recalls = df["recall"].apply(
                lambda x: x[positive_class_id]
            )

            plt.plot(
                recalls,
                precisions,
                marker="o",
                label=label,
            )

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim(0, 1)
        plt.gca().set_aspect("equal")
        plt.xlabel("Recall $\\frac{TP}{TP + FN}$", fontsize=14)
        plt.ylabel("Precision $\\frac{TP}{TP + FP}$", fontsize=14)
        plt.title(
            "Precision–Recall Curve Across Impostor Generation Techniques\n"
            f"Dataset: {dataset_name} ({positive_class})"
        )
        plt.legend()
        plt.tight_layout()

        fname = (
            f"roc_prec_recall_curve_{dataset_name.replace(' ', '_')}_"
            f"{positive_class.lower().replace(' ', '_')}.svg"
        )
        plt.savefig(
            LOCAL_SAVE_PATH / fname,
            bbox_inches="tight",
        )
        logger.info("Saved %s", fname)

        fig.clear()

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
    logging.info(f"Saved effectiveness scores as csv to {LOCAL_SAVE_PATH}/effectiveness_scores.csv.")

    plot_precision_recall_curve(results=results_dict, dataset_name=dataset_name)
    logging.info(
        f"Saved precision-recall plots as csv to {LOCAL_SAVE_PATH}/effectiveness_scores.csv."
    )
