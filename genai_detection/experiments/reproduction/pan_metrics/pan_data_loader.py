from __future__ import annotations

"""Load PAN predictions and ground truth from MongoDB."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import os
from bson import ObjectId
from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.experiments.reproduction.ablation_args import ABLATION_ARGS, ABLATION_DETECTORS
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB




@dataclass
class PANMethodConfig:
    """Configuration for loading method predictions from MongoDB."""

    name: str
    kind: str
    n_impostors: int | None = None
    rounds: int | None = None
    n_potential_impostors: int | None = None
    ablation_class_name: str | None = None
    retrieval_index: str | None = None
    impostor_technique: str | None = None


class PANDataLoader:
    """
    Load predictions and labels for PAN metrics from MongoDB.
    """

    def __init__(self, mongo: ParaphraseMongoDB | None = None, batch_size: int = 250) -> None:
        self.mongo = mongo or ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
        self.batch_size = batch_size

    def load_scores(
        self,
        method_name: str,
        dataset_name: str,
        *,
        impostor_technique: str | None = None,
        n_impostors: int = 50,
        n_potential_impostors: int | None = None,
        rounds: int = 100,
    ) -> Dict[Tuple[ObjectId, ObjectId], float]:
        """
        Load scores for a method name from the appropriate MongoDB collection.
        """
        if method_name in ABLATION_DETECTORS:
            return self._load_ablation_scores(
                variant=method_name,
                dataset_name=dataset_name,
                impostor_technique=impostor_technique,
            )

        if method_name in IMPOSTOR_GENERATORS:
            return self._load_impostor_scores(
                technique=method_name,
                dataset_name=dataset_name,
                n_impostors=n_impostors,
                n_potential_impostors=n_potential_impostors,
                rounds=rounds,
            )

        else:
            return self.load_or_compute_baseline_scores(
                method_name=method_name,
                dataset_name=dataset_name
            )

    def _load_impostor_scores(
        self,
        technique: str,
        dataset_name: str,
        *,
        n_impostors: int,
        rounds: int,
        n_potential_impostors: int | None = None,
    ) -> Dict[Tuple[ObjectId, ObjectId], float]:
        if technique not in IMPOSTOR_GENERATORS and "on_the_fly" not in technique:
            raise ValueError(f"Unsupported impostor technique: {technique}")

        # on the fly impostor generation uses different indices, which are stored in mongodb collection
        if "on_the_fly" in technique and technique != "on_the_fly":
            index = CONFIG.RETRIEVAL_INDEX_TRANSLATIONS.get(technique)
            if index is None:
                raise ValueError(f"Unknown retrieval index for {technique}.")
            technique = "on_the_fly"
        else:
            index = None

        query = {
            "impostor_generation_technique": technique,
            "n_impostors": n_impostors,
            "dataset_name": dataset_name,
        }
        if index is not None:
            query["retrieval_index"] = index
        if n_potential_impostors is not None:
            query["n_potential_impostors"] = n_potential_impostors

        cursor = self.mongo.impostor_output_collection.find(
            query,
            {"left_id": 1, "right_id": 1, "scores_over_different_rounds": 1},
            batch_size=self.batch_size,
        ).sort("_id", 1)

        scores_by_pair: Dict[Tuple[ObjectId, ObjectId], float] = {}
        for doc in cursor:
            left_id = doc["left_id"]
            right_id = doc["right_id"]
            pair = (left_id, right_id)
            if pair not in scores_by_pair:
                scores_by_pair[pair] = doc["scores_over_different_rounds"] / rounds

        return scores_by_pair

    def load_or_compute_baseline_scores(
        self,
        method_name: str,
        dataset_name: str,
    ) -> Dict[Tuple[ObjectId, ObjectId], float]:
        from genai_detection.experiments.reproduction.impostor_metrics import (
            load_all_pairs,
        )
        from genai_detection.experiments.reproduction.prec_recall_curves import (
            _build_baselines,
            _score_baselines_for_pair_batches,
        )

        baselines = _build_baselines(dataset_name=dataset_name)
        baseline = baselines.get(method_name)
        if baseline is None:
            raise ValueError(f"Unsupported baseline method: {method_name}")

        text_id_pairs, _ = load_all_pairs(dataset_name)
        if not text_id_pairs:
            return {}
        if len(text_id_pairs) % 2 != 0:
            raise ValueError(
                f"Flattened text list must contain even number of elements but length is {len(text_id_pairs)}"
            )

        text_test_pairs = self.mongo.get_texts_for_ids(text_ids=text_id_pairs)
        text_pairs = list(zip(text_test_pairs[0::2], text_test_pairs[1::2]))
        id_pairs = list(zip(text_id_pairs[0::2], text_id_pairs[1::2]))

        predictions = _score_baselines_for_pair_batches(
            baselines={method_name: baseline},
            pair_batches=[text_pairs],
            pair_id_batches=[id_pairs],
            dataset_name=dataset_name,
            mongoDB=self.mongo,
        )
        scores = predictions.get(method_name, [])
        if len(scores) != len(id_pairs):
            raise ValueError(
                f"Baseline {method_name}: expected {len(id_pairs)} scores, got {len(scores)}"
            )

        return {
            (ObjectId(left_id), ObjectId(right_id)): score
            for (left_id, right_id), score in zip(id_pairs, scores)
        }

    def _load_ablation_scores(
        self,
        variant: str,
        dataset_name: str,
        *,
        impostor_technique: str | None,
    ) -> Dict[Tuple[ObjectId, ObjectId], float]:
        if impostor_technique is None:
            raise ValueError("impostor_technique must be provided for ablation methods.")
        if variant not in ABLATION_ARGS:
            raise ValueError(f"Unknown ablation variant: {variant}")

        args = ABLATION_ARGS[variant]
        rounds = args["rounds"]
        n_impostors = args.get("n_impostors") or args.get("impostors_per_round")
        if n_impostors is None:
            raise ValueError(f"Missing n_impostors for ablation '{variant}'.")

        technique = impostor_technique
        query = {
            "impostor_generation_technique": technique,
            "n_impostors": n_impostors,
            "ablation": ABLATION_DETECTORS[variant].__name__,
            "dataset_name": dataset_name,
        }
        if "on_the_fly" in technique and technique != "on_the_fly":
            index = CONFIG.RETRIEVAL_INDEX_TRANSLATIONS.get(technique)
            if index is None:
                raise ValueError(f"Unknown retrieval index for {technique}.")
            query["impostor_generation_technique"] = "on_the_fly"
            query["retrieval_index"] = index

        cursor = self.mongo.impostor_ablation_output_collection.find(
            query,
            {"left_id": 1, "right_id": 1, "scores_over_different_rounds": 1},
            batch_size=self.batch_size,
        ).sort("_id", 1)

        scores_by_pair: Dict[Tuple[ObjectId, ObjectId], float] = {}
        for doc in cursor:
            pair = (doc["left_id"], doc["right_id"])
            if pair not in scores_by_pair:
                scores_by_pair[pair] = doc["scores_over_different_rounds"] / rounds

        return scores_by_pair

    def load_ground_truth(
        self,
        pairs: List[Tuple[ObjectId, ObjectId]],
        dataset_name: str,
    ) -> Dict[Tuple[ObjectId, ObjectId], int]:
        gt_by_pair: Dict[Tuple[ObjectId, ObjectId], int] = {}
        for batch in self._iter_batches(pairs, self.batch_size):
            or_conditions = [{"left_id": l, "right_id": r} for l, r in batch]
            cursor = self.mongo.all_pairs_collection.find(
                {"dataset_name": dataset_name, "$or": or_conditions},
                {"left_id": 1, "right_id": 1, "same": 1},
                batch_size=self.batch_size,
            )
            for doc in cursor:
                gt_by_pair[(doc["left_id"], doc["right_id"])] = int(doc["same"])
        return gt_by_pair

    @staticmethod
    def _iter_batches(items: List, batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]
