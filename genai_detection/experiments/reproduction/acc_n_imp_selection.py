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
from genai_detection.experiments.reproduction.impostor_metrics import (
    load_test_pairs,
    compute_metrics_for_thresholds,
    extract_best_metric,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[3]
    / CONFIG.SAVE_PATH
    / "reproduction"
    / "n_imp_selection_config"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def compute_acc_across_n_selected_potential_imps(
    dataset_name: str,
    imp_gen_techniques: List[str],
) -> Dict[str, Dict[int, Dict[int, pd.DataFrame]]]:
    """
    Reproduction of Tables 1 & 2 from Koppel et al. (2014).

    Returns:
        results[technique][n_selected][n_potential] -> pd.DataFrame
    """

    logger.info(
        "Running n-impostor selection experiment on %s with %s",
        dataset_name,
        imp_gen_techniques,
    )

    assert all(
        t in IMPOSTOR_GENERATORS for t in imp_gen_techniques
    ), f"Unsupported impostor generator in {imp_gen_techniques}"

    # TODO: Currently reported in the paper
    imp_gen_techniques = ["on_the_fly", "in_domain"]

    N_IMP_SELECTION_CONFIG = {
        "n_selected": [10, 25, 50, 100],
        "n_potential": [100, 250, 500, 1000],
    }

    # Load test data once
    text_test_pairs, ground_truth = load_test_pairs(dataset_name)

    results: Dict[str, Dict[int, Dict[int, pd.DataFrame]]] = {}

    for technique in imp_gen_techniques:
        logger.info("Processing technique: %s", technique)
        results[technique] = {}

        for n_selected in N_IMP_SELECTION_CONFIG["n_selected"]:
            detector = ImpostorDetector(
                impostor_technique=technique,
                n_impostors=n_selected,
                dataset_name=dataset_name,
            )
            results[technique][n_selected] = {}

            for n_potential in N_IMP_SELECTION_CONFIG["n_potential"]:
                detector.impostor_generator.set_num_potential_impostors(
                    num_potential_impostors=n_potential
                )

                scores = detector.get_score(text=text_test_pairs)
                assert scores is not None, "Detector returned None scores"

                df_metrics = compute_metrics_for_thresholds(
                    ground_truth,
                    scores,
                )

                results[technique][n_selected][n_potential] = df_metrics

                logger.info(
                    "Done %s | n_selected=%d | n_potential=%d",
                    technique,
                    n_selected,
                    n_potential,
                )

    return results


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_acc_curve(results: Dict, dataset_name: str):
    """
    Visualizes results as heatmaps (accuracy, f1, precision, recall).
    """

    METRICS = ["f1", "accuracy", "precision", "recall"]

    for positive_class_id in [0, 1]:
        positive_class = (
            "Same Author" if positive_class_id == 1 else "Different Author"
        )

        for technique, nested_dict in results.items():
            tech_name = CONFIG.LABEL_TRANSLATIONS.get(technique, technique)

            n_selected_vals = sorted(nested_dict.keys())
            n_potential_vals = sorted(
                next(iter(nested_dict.values())).keys()
            )

            for metric in METRICS:
                if metric == "accuracy" and positive_class_id == 0:
                    continue

                heat = np.zeros(
                    (len(n_potential_vals), len(n_selected_vals))
                )

                for i, n_pot in enumerate(n_potential_vals):
                    for j, n_sel in enumerate(n_selected_vals):
                        df = nested_dict[n_sel][n_pot]
                        heat[i, j] = extract_best_metric(df=df,metric=metric, positive_class_id=positive_class_id)

                # ---------- Plot ----------
                plt.figure(figsize=(8, 6))
                plt.imshow(heat, aspect="auto")
                plt.colorbar(label=f"Max {metric}")
                plt.xticks(range(len(n_selected_vals)), n_selected_vals)
                plt.yticks(range(len(n_potential_vals)), n_potential_vals)
                plt.xlabel("Number of Selected Impostors")
                plt.ylabel("Number of Potential Impostors")
                plt.title(
                    f"{tech_name} — Max {metric} "
                    f"({dataset_name}, {positive_class})"
                )

                save_path = LOCAL_SAVE_PATH / technique
                save_path.mkdir(parents=True, exist_ok=True)

                fname = (
                    f"heatmap_{technique}_{metric}_"
                    f"{positive_class.replace(' ', '_')}_"
                    f"{dataset_name.replace(' ', '_')}.svg"
                )
                plt.savefig(save_path / fname, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved %s", fname)

                # ---------- CSV ----------
                csv_name = fname.replace(".svg", ".csv")

                df_csv = pd.DataFrame(
                    heat,
                    index=n_potential_vals,
                    columns=n_selected_vals,
                ).T
                df_csv.index.name = "n_potential"
                df_csv.columns.name = "n_selected"

                df_csv.to_csv(save_path / csv_name)
                logger.info("Saved %s", csv_name)


def run_acc_curves(dataset_name:str, imp_gen_techniques:List[str]):
    results_dict = compute_acc_across_n_selected_potential_imps(
        dataset_name=dataset_name, imp_gen_techniques=imp_gen_techniques
    )

    logging.info(f"results_dict: {results_dict}")
    plot_acc_curve(results=results_dict, dataset_name=dataset_name)
