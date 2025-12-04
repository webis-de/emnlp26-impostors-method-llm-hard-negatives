# Copyright 2024 Klara M. Gutekunst, Webis
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.detectors.impostor_supervised_baseline import SupervisedImpostorBaseline
from genai_detection.detectors.impostor_unsupervised_baseline import UnSupervisedImpostorBaseline
from genai_detection.detectors.ppmd import PPMdDetector
from genai_detection.detectors.unmasking import UnmaskingDetector
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )


LOCAL_SAVE_PATH = Path(__file__).resolve().parents[3] / CONFIG.SAVE_PATH / "reproduction"
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)


# reproduction of Figure 4 a, b from Koppel et al. (2014)
def compute_prec_recall_f1_acc_dict(
    dataset_name:str,
    imp_gen_techniques: List[str],
) -> Dict[str, pd.DataFrame]:
    """

    :param dataset_name: The name of the dataset to use.
    :param imp_gen_techniques: List of impostor generation techniques to use.
    """
    logging.info(
        "Reproducing Figure 4 with different impostor generation techniques: %s",
        imp_gen_techniques,
    )
    mongoDB = ParaphraseMongoDB()

    assert all([appr in IMPOSTOR_GENERATORS.keys() for appr in imp_gen_techniques]), (f"Impostor generation technique "
                                                                                      f"{imp_gen_techniques} not "
                                                                                      f"supported. Supported are: {IMPOSTOR_GENERATORS.keys()}")

    # get IDs and ground truth of text pairs from the "text_pair" mongoDB collection
    test_pairs = list(mongoDB.find_document_by_non_id_field(collection=mongoDB.test_pairs_collection,
                                                            document_field_name="dataset_name",document_value=dataset_name))
    logging.info(f"test_pairs: {test_pairs}")
    text_test_pairs = [str(id_) for pair in test_pairs for id_ in (pair["left_id"], pair["right_id"])]
    ground_truth_test_pairs = [pair["same"] for pair in test_pairs]
    logging.info("Number of text pairs in test dataset: %d", len(text_test_pairs))
    logging.info(f"text pairs:\n{text_test_pairs}")

    # get results for text pairs (given as IDs) for each of the impostor generation options
    predictions = {"ground_truth":ground_truth_test_pairs}
    for imp_generation_technique in imp_gen_techniques:
        logging.info(f"Start obtaining impostor scores with {imp_generation_technique}...")
        # TODO: use more impostors
        impostor_detector = ImpostorDetector(impostor_technique=imp_generation_technique, n_impostors=2)
        # if existent, pre-computed scores are used
        score = impostor_detector.get_score(text=text_test_pairs)
        assert score is not None, f"{imp_generation_technique} returned None."
        predictions[imp_generation_technique] = score
        logging.info(f"Finished obtaining impostor scores with {imp_generation_technique}.\n")
    logging.info(f"Finished obtaining impostor scores.\n{predictions}\n")
    # FIXME:
    baselines = {
        "unsupervised_baseline_min-max": UnSupervisedImpostorBaseline(
            use_cosine_simiarity=False, dataset_name=dataset_name
        ),
        "unsupervised_baseline_cosine": UnSupervisedImpostorBaseline(
            use_cosine_simiarity=True, dataset_name=dataset_name
        ),
        "supervised_baseline": SupervisedImpostorBaseline(dataset_name=dataset_name),
        "unmasking": UnmaskingDetector(),
        "ppmd": PPMdDetector(),
    }
    for baseline_name, baseline in baselines.items():
        preds = baseline.get_score(text_test_pairs)
        preds = preds.tolist() if not isinstance(preds, list) else preds

        predictions[f"{baseline_name.replace(' ','_')}_score"] = np.array(
            preds
        ).ravel().tolist()

    # compute precision, recall, accuracy, f1 for 11-points
    thresholds = np.arange(0.0, 1.05, 0.1)

    def compute_metrics_for_approach(args:Tuple[str, List[int]]):
        """
        Compute metrics for a single approach across thresholds.
        args is a key (i.e., approach name) and a list of predictions (i.e., float).
        """
        imp_gen_technique, preds_of_imp_gen_technique = args
        rows = []
        for t in thresholds:
            binary_preds = [1 if p >= t else 0 for p in preds_of_imp_gen_technique]
            # average = None makes method return score for each class (i.e., same- and different-author)
            rows.append(
                {
                    "threshold": t,
                    "precision": precision_score(   # returns an array
                        ground_truth_test_pairs,
                        binary_preds,
                        zero_division=0,
                        average=None,
                    ),
                    "recall": recall_score(         # returns an array
                        ground_truth_test_pairs,
                        binary_preds,
                        zero_division=0,
                        average=None,
                    ),
                    "f1": f1_score(                 # returns an array
                        ground_truth_test_pairs,
                        binary_preds,
                        zero_division=0,
                        average=None,
                    ),
                    "accuracy": accuracy_score(     # returns a float (i.e., not array)
                        ground_truth_test_pairs, binary_preds, normalize=True
                    ),
                }
            )
        return imp_gen_technique, pd.DataFrame(rows)

    args_list = [(key, preds) for key, preds in predictions.items() if key != "ground_truth"]

    # Compute in parallel
    results = {}
    logging.info(f"Computing metrics for {len(args_list)} approaches in parallel...")
    with ThreadPoolExecutor() as executor:
        for key, df in executor.map(compute_metrics_for_approach, args_list):
            results[key] = df
            logging.info(f"Computed metrics for {key}.")
    logging.info("Finished computing metrics.")

    return results


def plot_precision_recall_curve(results:dict, dataset_name:str):
    """
    Visualizes the impostor detection results via Precision-Recall curves for different impostor generation techniques (cf. Figures 4 a, b from Koppel et al. (2014)).

    :param results: Dictionary of impostor detection results. Keys are impostor generation techniques, values are
    dataframe scores. Each dataframe has eleven rows (i.e., for different thresholds), and the columns 'threshold',
    'precision', 'recall', 'f1', and 'accuracy'. All cells but those of threshold and accuracy contain arrays of two
    values. The first value for the positive class being 0 (i.e., for different-author), and the second value for the
    positive class being 1 (i.e., same-author).
    """
    label_translations = {
        "in_domain": "In-Domain",
        "on_the_fly": "On-the-Fly",
        "naive_llm": "One-Step Paraphraser (LLM)",
        "non_naive_llm": "Two-Step Paraphraser (LLM)",
        "unsupervised_baseline_min-max": "Unsup. Min-Max (B)",
        "unsupervised_baseline_cosine": "Unsup. Cosine (B)",
        "supervised_baseline": "Sup. SVM (B)",
        "unmasking": "Unmasking",
        "ppmd": "PPMd",
    }

    fig = plt.figure(figsize=(10, 7))

    for positive_class_id in [0, 1]:
        positive_class = "Same Author" if positive_class_id == 1 else "Different Author"
        for key, df in results.items():
            # Pick translated label if available
            label = label_translations.get(key, key)

            # Extract precision/recall for positive class
            precisions = df["precision"].apply(lambda x: x[positive_class_id]).values
            recalls = df["recall"].apply(lambda x: x[positive_class_id]).values

            # Plot PR curve
            plt.plot(recalls, precisions, marker='o', label=label)

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim(0, 1)
        plt.gca().set_aspect("equal")
        plt.xlabel("Recall $\\frac{{TP}}{{TP + FN}}$", fontsize=14)
        plt.ylabel("Precision $\\frac{{TP}}{{TP + FP}}$", fontsize=14)
        title = f"Precision-Recall Curve Across Impostor Generation Techniques\nOn {dataset_name} Data with {positive_class} Class"
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        for format in ["svg"]:  # "png",
            figure_name = (
                f"11111roc_prec_recall_curve_dif_{dataset_name.replace(' ', '_')}_"
                f"{positive_class.lower().replace(' ', '_')}_imp_gen"
                f".{format}"
            )
            plt.savefig(LOCAL_SAVE_PATH / figure_name, bbox_inches="tight")
            logging.info(f"Saved figure {figure_name} to {LOCAL_SAVE_PATH}.")
        fig.clear()
