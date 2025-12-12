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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )


LOCAL_SAVE_PATH = Path(__file__).resolve().parents[3] / CONFIG.SAVE_PATH / "reproduction" / "n_imp_selection_config"
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# reproduction of Tab. 1, 2 from Koppel et al. (2014)
def compute_acc_across_n_selected_potential_imps(
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
    # TODO: implement for more imp generator techniques
    imp_gen_techniques = ["in_domain"]

    N_IMP_SELECTION_CONFIG = {"n_selected": [10, 25, 50, 100], "n_potential": [100, 250, 500, 1000]}

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
        predictions[imp_generation_technique] = {}
        for n_impostors in N_IMP_SELECTION_CONFIG["n_selected"]:
            impostor_detector = ImpostorDetector(impostor_technique=imp_generation_technique, n_impostors=n_impostors)
            predictions[imp_generation_technique][n_impostors] = {}
            for n_potential in N_IMP_SELECTION_CONFIG["n_potential"]:
                impostor_detector.impostor_generator.set_num_potential_impostors(num_potential_impostors=n_potential)
                # if existent, pre-computed scores are used
                score = impostor_detector.get_score(text=text_test_pairs)
                assert score is not None, f"{imp_generation_technique} returned None."
                predictions[imp_generation_technique][n_impostors][n_potential] = score
        logging.info(f"Finished obtaining impostor scores with {imp_generation_technique}.\n")
    logging.info(f"Finished obtaining impostor scores.\n{predictions}\n")

    # compute precision, recall, accuracy, f1 for 11-points
    thresholds = np.arange(0.0, 1.05, 0.1)

    def compute_metrics_for_approach(
        args: Tuple[str, Dict[int, Dict[int, List[float]]]],
    ):
        """
        Compute metrics for a single approach across thresholds.
        args = (technique_name, nested_prediction_dict)
          where nested_prediction_dict[n_selected][n_potential] = list of scores
        """
        technique, nested_preds = args
        metrics = {}

        for n_selected, potential_dict in nested_preds.items():
            metrics[n_selected] = {}
            for n_potential, preds in potential_dict.items():

                rows = []
                for t in thresholds:
                    binary_preds = [1 if p >= t else 0 for p in preds]
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
                metrics[n_selected][n_potential] = pd.DataFrame(rows)
        return technique, metrics

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


def plot_acc_curve(results:dict, dataset_name:str):
    """
    Visualizes the impostor detection results via heatmaps of different effectiveness measures for different impostor
    generation
    techniques (cf.
    Tab. 1, 2 from Koppel et al. (2014)).

    :param results: Dictionary of impostor detection results. Keys are impostor generation techniques, values are
    nested dataframes. The next key indicates the number of selected impostors. The value is a dataframe, 
    where the key is the number of potential impostors and the values dataframe of scores: Each dataframe has eleven
    rows (i.e., for different thresholds), and the columns, 'threshold', 'precision', 'recall', 'f1', and 'accuracy'.
    All cells but those of threshold and accuracy contain arrays of two values. The first value for the positive
    class being 0 (i.e., for different-author), and the second value for the positive class being 1 (i.e., same-author).
    """
    label_translations = {
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

    METRICS = ["f1", "accuracy", "precision", "recall"]

    for positive_class_id in [0, 1]:
        positive_class = "Same Author" if positive_class_id == 1 else "Different Author"
        for technique, nested_dict in results.items():
            tech_name = label_translations.get(technique, technique)

            # extract grid dimensions
            n_selected_vals = sorted(nested_dict.keys())
            n_potential_vals = sorted(list(next(iter(nested_dict.values())).keys()))

            # prepare matrix per metric
            for metric in METRICS:

                heat = np.zeros((len(n_potential_vals), len(n_selected_vals)))
                if metric == "accuracy" and positive_class_id == 0:
                    continue

                for i, n_pot in enumerate(n_potential_vals):
                    for j, n_sel in enumerate(n_selected_vals):
                        df = nested_dict[n_sel][n_pot]

                        if metric == "accuracy":
                            # accuracy is a float per threshold → take max directly
                            best = df["accuracy"].max()
                        else:
                            # f1 / precision / recall return arrays → use second entry (=positive class = same-author)
                            best = df[metric].apply(lambda arr: arr[positive_class_id]).max()

                        heat[i, j] = best

                # plot heatmap
                plt.figure(figsize=(8, 6))
                plt.imshow(heat, aspect="auto")
                plt.colorbar(label=f"Max {metric}")
                plt.xticks(range(len(n_selected_vals)), n_selected_vals)
                plt.yticks(range(len(n_potential_vals)), n_potential_vals)
                plt.xlabel("Number of Selected Impostors")
                plt.ylabel("Number of Potential Impostors")
                plt.title(f"{tech_name} — Max {metric} Heatmap ({dataset_name}, {positive_class})")

                # ---------- save plot ----------
                save_path = LOCAL_SAVE_PATH / technique
                save_path.mkdir(parents=True, exist_ok=True)
                fname = (
                    f"heatmap_{technique}_{metric}_"
                    f"{positive_class.replace(' ', '_')}_{dataset_name.replace(' ', '_')}.svg"
                )
                plt.savefig(save_path / fname, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Saved {fname}.")
                # ---------- save CSV ----------
                csv_name = (
                    f"heatmap_{technique}_{metric}_"
                    f"{positive_class.replace(' ', '_')}_{dataset_name.replace(' ', '_')}.csv"
                )

                df_csv = pd.DataFrame(
                    heat,
                    index=n_potential_vals,
                    columns=n_selected_vals,
                )
                df_csv = df_csv.T
                df_csv.index.name = "n_potential"
                df_csv.columns.name = "n_selected"

                df_csv.to_csv(save_path / csv_name)
                logging.info(f"Saved {csv_name}.")
