"""
The purpose of this file is to run the experiment where we use different training data configurations to optimize the
supervised baseline from Koppel et al. (2014).
- Leave One Out (i.e., train on all but "one" input pair and rotate which pair is considered for test)
    - with leaving out all pairs which contain any of two authors from test pair in training data
    - c.f. above but only train on pairs from the same two assignments
Training data should be balanced.
"""
import logging
import os
from abc import ABC
from pathlib import Path
from typing import List, Iterable

import numpy as np
import pandas as pd

from genai_detection.config import CONFIG
from genai_detection.detectors.impostor_supervised_baseline import SupervisedImpostorBaseline
from genai_detection.experiments.reproduction.impostor_metrics import compute_metrics_for_binary_predictions, \
    compute_metrics_for_thresholds, extract_best_metric
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[3]
    / CONFIG.SAVE_PATH
    / "supervised_baseline"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

class OptimalSupervisedBaseline(ABC):
    def __init__(self):
        self.mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
        self.all_pairs_collection = self.mongoDB.all_pairs_collection
        logger.info(f"Obtained pairs from mongoDB.")

    def train_svc(self, dataset_name, test_pair, additional_in_args:List[str]):
        return SupervisedImpostorBaseline(
            dataset_name=dataset_name,
            left_input={key:val for key, val in test_pair.items() if "left" in key},
            right_input={key:val for key, val in test_pair.items() if "right" in key},
            additional_in_args=additional_in_args,
        )


    def obtain_loo_preds_gt_for_one_config(self, config:List[str], dataset_name:str=CONFIG.STUDENT_ESSAYS,):
        predictions = {}
        gt = {}
        for test_pair in self.mongoDB.find_document_by_non_id_field(
                collection=self.all_pairs_collection,
                document_field_name="dataset_name",
                document_value=dataset_name,
        ):
            pair_id = test_pair["_id"]
            gt[pair_id] = test_pair["same"]
            search_args = {"pair_id": pair_id, "config": config, "dataset_name": dataset_name}
            pred_cursor = self.mongoDB.find_document_by_multiple_fields(
                collection=self.mongoDB.supervised_baseline_diff_config_preds_collection,
                search_args= search_args,
            )
            existing_result = list(pred_cursor)
            if not existing_result:
                assert "left_id" in test_pair and "right_id" in test_pair, f"text IDs of test pair missing; only columns: {test_pair.keys()}"
                sup_baseline = self.train_svc(dataset_name=test_pair["dataset_name"], test_pair=test_pair, additional_in_args=config)
                text_test_pairs = self.mongoDB.get_texts_for_ids(text_ids=[test_pair["left_id"], test_pair["right_id"]])

                preds = sup_baseline.get_score(text_test_pairs)
                preds = preds.tolist() if hasattr(preds, "tolist") else preds
                predictions[pair_id] = np.asarray(preds).ravel().tolist()[0]
                self.mongoDB.insert_document(
                    collection=self.mongoDB.supervised_baseline_diff_config_preds_collection,
                    insert_data={
                        **search_args,
                        "prediction": predictions[pair_id],
                        "ground_truth": gt[pair_id],
                    }
                )
            else:
                predictions[pair_id] = existing_result[0]["prediction"]
        logger.info(f"Obtained {len(predictions)} predictions for pairs of dataset {dataset_name}.")
        return predictions, gt

    def run_experiment(
            self
    ):
        dataset_names = [CONFIG.STUDENT_ESSAYS, CONFIG.BLOG]
        configs = [None, ["assignment"]]
        scores = pd.DataFrame()
        for dataset_name in dataset_names:
            for config in configs:
                preds, gt = self.obtain_loo_preds_gt_for_one_config(config=config, dataset_name=dataset_name)
                # compute recall, precision, f1, accuracy for different thresholds
                sup_preds = list(preds.values())
                df_metrics = compute_metrics_for_thresholds(
                    ground_truth=list(gt.values()),
                    scores=sup_preds,
                    thresholds=np.arange(np.min(sup_preds), np.max(sup_preds), 0.01),
                )
                metrics = ["precision", "recall", "f1", "accuracy"]
                positive_class_translation = ["Different Author", "Same Author"]

                scores_for_config = {
                    positive_class_translation[positive_class_id]: {
                        metric: extract_best_metric(
                            df=df_metrics,
                            metric=metric,
                            positive_class_id=positive_class_id
                        )
                        for metric in metrics
                    }
                    for positive_class_id in [0, 1]
                }
                for positive_class, class_score_config_df in scores_for_config.items():
                    class_score_config_df[dataset_name] = dataset_name
                    class_score_config_df["config"] = config if config else ""
                    self.mongoDB.insert_document(
                        collection=self.mongoDB.supervised_baseline_diff_config_scores_collection,
                        insert_data=class_score_config_df)
                    scores = pd.concat([scores, class_score_config_df])
                logger.info(f"Obtained {len(scores)} scores for dataset {dataset_name} and config {config}.")

        # save dataframe to disk
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


