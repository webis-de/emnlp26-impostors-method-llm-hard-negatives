# Copyright 2026 Klara M. Gutekunst, Webis
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

"""
Run the supervised-baseline optimization experiment from Koppel et al. (2014).

What happens end-to-end:
1. For each dataset and configuration, iterate over all author pairs in MongoDB.
2. For each pair, reuse an existing prediction if stored; otherwise train a supervised
   impostor baseline in a leave-one-out fashion and compute a score.
3. Persist per-pair predictions to MongoDB, then compute thresholded metrics
   (precision/recall/F1/accuracy) for both "same" and "different" author labels.
4. Persist per-config scores to MongoDB and write a CSV summary to disk.

Notes:
- Leave-one-out training excludes the current test pair and, depending on the config,
  may also exclude any pair sharing either author.
- The training data is expected to be balanced; invalid or missing predictions are skipped.
"""
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from genai_detection.config import CONFIG
from genai_detection.detectors.impostor_supervised_baseline import SupervisedImpostorBaseline
from genai_detection.experiments.reproduction.impostor_metrics import compute_metrics_for_thresholds, \
    extract_best_metric_and_position
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

# ray.init()

# ray job submit --address https://ray.srv.webis.de --working-dir . --runtime-env env.yml -- python genai_detection/experiments/opt_sup_baseline.py

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "supervised_baseline"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)


class OptimalSupervisedBaseline:
    """Orchestrates the supervised-baseline optimization and persistence of results."""

    def __init__(self):
        """Initialize MongoDB connections and relevant collections."""
        self.mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
        self.all_pairs_collection = self.mongoDB.all_pairs_collection
        logger.info(f"Obtained pairs from mongoDB.")

    @staticmethod
    def train_svc(dataset_name, test_pair, additional_in_args:List[str]):
        """Create a SupervisedImpostorBaseline configured for a specific test pair."""
        return SupervisedImpostorBaseline(
            dataset_name=dataset_name,
            left_input={key:val for key, val in test_pair.items() if "left" in key},
            right_input={key:val for key, val in test_pair.items() if "right" in key},
            additional_in_args=additional_in_args,
        )



    def obtain_loo_preds_gt_for_one_config(self, config:List[str], dataset_name:str=CONFIG.STUDENT_ESSAYS,):
        """
        Obtain leave-one-out predictions and ground-truth labels for a config.

        This method:
        - checks for stored predictions in MongoDB,
        - trains a baseline if needed,
        - stores fresh predictions back to MongoDB, and
        - returns dicts keyed by pair id.
        """
        predictions = {}
        gt = {}
        for test_pair in self.mongoDB.find_document_by_non_id_field(
                collection=self.all_pairs_collection,
                document_field_name="dataset_name",
                document_value=dataset_name,
        ):
            pair_id = test_pair["_id"]
            search_args = {"pair_id": pair_id, "config": config, "dataset_name": dataset_name}
            # Prefer cached predictions if available to avoid retraining.
            pred_cursor = self.mongoDB.find_document_by_multiple_fields(
                collection=self.mongoDB.supervised_baseline_diff_config_preds_collection,
                search_args= search_args,
            )
            existing_result = list(pred_cursor)
            pred_doc = existing_result[0] if existing_result else None
            pred_val = pred_doc.get("prediction") if pred_doc else None
            if pred_val is None or (isinstance(pred_val, float) and np.isnan(pred_val)):
                assert "left_id" in test_pair and "right_id" in test_pair, f"text IDs of test pair missing; only columns: {test_pair.keys()}"
                if config:
                    # logger.info(f"About to use test pair keys: {test_pair.keys()}.")
                    for config_item in config:
                        assert f"left_{config_item}" in test_pair and f"right_{config_item}" in test_pair, f"Required keys not found in test pair: {test_pair.keys()}"
                try:
                    # Train on all other pairs according to the config (leave-one-out).
                    sup_baseline = self.train_svc(dataset_name=test_pair["dataset_name"], test_pair=test_pair, additional_in_args=config)
                    if sup_baseline is None:
                        logger.error(f"Could not obtain prediction for test pair {pair_id} due to OOM when training.")
                        # avoid OOM
                        continue
                    if sup_baseline.train_dataset.shape[0] == 0:
                        logger.error(f"Could not train baseline for test pair {pair_id} due to no existing training data.")
                        continue
                except ValueError as e:
                    logger.error(f"Skipping this test pair because SVC training data contains only one class. Error: {e}")
                    continue
                # Fetch the actual text content for scoring.
                text_test_pairs = self.mongoDB.get_texts_for_ids(text_ids=[test_pair["left_id"], test_pair["right_id"]])

                preds = sup_baseline.get_score(text_test_pairs)
                preds = preds.tolist() if hasattr(preds, "tolist") else preds
                pred_scalar = np.asarray(preds).ravel().tolist()[0]
                if pred_scalar is None or (isinstance(pred_scalar, float) and np.isnan(pred_scalar)):
                    logger.error(f"Obtained invalid prediction for test pair {pair_id}; skipping insert.")
                    continue
                predictions[pair_id] = pred_scalar
                # Persist the freshly computed prediction for reuse.
                self.mongoDB.insert_document(
                    collection=self.mongoDB.supervised_baseline_diff_config_preds_collection,
                    insert_data={
                        **search_args,
                        "prediction": predictions[pair_id],
                        "ground_truth": test_pair["same"],
                    }
                )
            else:
                predictions[pair_id] = pred_val
            gt[pair_id] = test_pair["same"] # only called if test pair is not skipped to avoid unequal length of gt and predictions
        logger.info(f"Obtained {len(predictions)} predictions for {len(gt)} pairs of dataset {dataset_name}.")
        if len(predictions) != len(gt):
            logger.error("Number of predictions does not match number of gt pairs.")
            logger.error(f"Symmetric key difference: {sorted(set(predictions.keys()) ^ set(gt.keys()))}")
        return predictions, gt

    def run_experiment(
            self
    ):
        """Run the optimization across datasets/configs and persist scores."""
        dataset_names = [CONFIG.STUDENT_ESSAYS, CONFIG.BLOG]
        configs = [None, ["assignment"]]
        scores = pd.DataFrame()
        for dataset_name in dataset_names:
            for config in configs:
                # Gather per-pair predictions and ground-truth labels.
                preds, gt = self.obtain_loo_preds_gt_for_one_config(config=config, dataset_name=dataset_name)
                # compute recall, precision, f1, accuracy for different thresholds
                invalid_pred_keys = [
                    k for k, v in preds.items()
                    if v is None or (isinstance(v, float) and np.isnan(v))
                ]
                if invalid_pred_keys:
                    logger.warning(f"Found {len(invalid_pred_keys)} invalid predictions; excluding from metrics.")
                    for k in invalid_pred_keys:
                        preds.pop(k, None)
                        gt.pop(k, None)
                assert len(preds) == len(gt), f"{len(preds)} != {len(gt)} (length of predictions and ground truths)"
                if not preds:
                    logger.error(f"No valid predictions for dataset {dataset_name} and config {config}; skipping.")
                    continue
                # Sweep thresholds across the prediction range.
                sup_preds = list(preds.values())
                sup_thresholds = np.arange(np.min(sup_preds), np.max(sup_preds), 0.01)
                df_metrics = compute_metrics_for_thresholds(
                    ground_truth=list(gt.values()),
                    scores=sup_preds,
                    thresholds=sup_thresholds,
                )
                metrics = ["precision", "recall", "f1", "accuracy"]
                positive_class_translation = ["Different Author", "Same Author"]

                rows = []

                for positive_class_id in [0, 1]:
                    class_name = positive_class_translation[positive_class_id]

                    flat = {
                        "dataset_name": dataset_name,
                        "config": config if config else "",
                        "positive_class": class_name
                    }

                    for metric in metrics:
                        best_score = extract_best_metric_and_position(df=df_metrics, metric=metric, positive_class_id=positive_class_id)
                        flat[f"{metric}_best_score"] = best_score["best_score"]
                        flat[f"{metric}_best_threshold"] = sup_thresholds[best_score["best_idx"]]

                    rows.append(flat)

                scores_config_df = pd.DataFrame(rows)

                # Insert each row separately to satisfy expected dict input.
                for doc in rows:
                    self.mongoDB.insert_document(
                        collection=self.mongoDB.supervised_baseline_diff_config_scores_collection,
                        insert_data=doc
                    )

                scores = pd.concat([scores, scores_config_df])
                logger.info(f"Obtained (same- and different-author) scores for dataset {dataset_name} and config {config}.")

        # Save the aggregated metrics to disk for offline inspection.
        scores.to_csv(LOCAL_SAVE_PATH / "effectiveness_scores.csv", index=False)
        logger.info(f"Saved effectiveness scores as csv to {LOCAL_SAVE_PATH}/effectiveness_scores.csv.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    optimal_sup_baseline = OptimalSupervisedBaseline()
    optimal_sup_baseline.run_experiment()
