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
Precision-recall curves for one-step LLM impostors without generating new impostors.

This entrypoint evaluates only pairs that either already have a one_step_llm score
in CONFIG.MONGO_IMPOSTOR_OUTPUT_COLLECTION or can be scored from existing
one-step paraphrases in CONFIG.MONGO_NAIVE_PARAPHRASE_COLLECTION. The detector is
called only with n_impostors values that are already available for both texts in a
pair, so NaiveImpostorGenerator returns MongoDB paraphrases instead of generating
new ones.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.experiments.reproduction.impostor_metrics import (
    compute_metrics_for_thresholds,
)
from genai_detection.experiments.reproduction.prec_recall_curves import (
    _build_baselines,
    _score_baselines_for_pair_batches,
    run_prec_recall_curves,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

ONE_STEP_LLM_TECHNIQUE = "one_step_llm"

class NoEligiblePairsError(Exception):
    """Raised when no eligible pairs are available for evaluation."""


def _to_object_id(value) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)


def _iter_batches(items: List, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _count_naive_paraphrases(
    mongo_db: ParaphraseMongoDB,
    text_ids: Iterable[ObjectId],
) -> Dict[ObjectId, int]:
    text_ids = list(text_ids)
    if not text_ids:
        return {}

    pipeline = [
        {"$match": {"text_id": {"$in": text_ids}}},
        {"$group": {"_id": "$text_id", "count": {"$sum": 1}}},
    ]
    return {
        _to_object_id(doc["_id"]): int(doc["count"])
        for doc in mongo_db.naive_paraphrase_collection.aggregate(
            pipeline,
            allowDiskUse=True,
        )
    }


def _load_existing_one_step_scores(
    mongo_db: ParaphraseMongoDB,
    dataset_name: str,
    rounds: int,
    n_impostors: Optional[int],
    n_potential_impostors: Optional[int],
) -> Dict[Tuple[ObjectId, ObjectId], float]:
    query = {
        "dataset_name": dataset_name,
        "impostor_generation_technique": ONE_STEP_LLM_TECHNIQUE,
    }
    if n_impostors is not None:
        query["n_impostors"] = n_impostors
    if n_potential_impostors is not None:
        query["n_potential_impostors"] = n_potential_impostors

    cursor = mongo_db.impostor_output_collection.find(
        query,
        {"left_id": 1, "right_id": 1, "scores_over_different_rounds": 1},
    ).sort("_id", 1)

    scores_by_pair: Dict[Tuple[ObjectId, ObjectId], float] = {}
    for doc in cursor:
        pair = (_to_object_id(doc["left_id"]), _to_object_id(doc["right_id"]))
        scores_by_pair[pair] = float(doc["scores_over_different_rounds"]) / rounds

    logger.info(
        "Loaded %d existing one_step_llm scores from %s for dataset '%s'.",
        len(scores_by_pair),
        CONFIG.MONGO_IMPOSTOR_OUTPUT_COLLECTION,
        dataset_name,
    )
    return scores_by_pair


def _load_pair_docs(
    mongo_db: ParaphraseMongoDB,
    dataset_name: str,
) -> List[dict]:
    pair_docs = list(
        mongo_db.all_pairs_collection.find(
            {"dataset_name": dataset_name},
            {"left_id": 1, "right_id": 1, "same": 1},
        ).sort("_id", 1)
    )
    logger.info("Loaded %d all-pairs documents for dataset '%s'.", len(pair_docs), dataset_name)
    return pair_docs


def _score_missing_pairs_from_existing_naive_paraphrases(
    grouped_pairs: Dict[int, List[Tuple[ObjectId, ObjectId]]],
    dataset_name: str,
    batch_size_pairs: int,
    rounds: int,
    top_n: int,
    portion_delete: float,
    upsample: bool,
) -> Dict[Tuple[ObjectId, ObjectId], float]:
    computed_scores: Dict[Tuple[ObjectId, ObjectId], float] = {}

    for n_impostors, pairs in sorted(grouped_pairs.items()):
        detector = ImpostorDetector(
            rounds=rounds,
            top_n=top_n,
            portion_delete=portion_delete,
            n_impostors=n_impostors,
            impostor_technique=ONE_STEP_LLM_TECHNIQUE,
            dataset_name=dataset_name,
            upsample=upsample,
        )
        logger.info(
            "Scoring %d missing pairs with %d existing one-step paraphrases per side.",
            len(pairs),
            n_impostors,
        )

        for batch in _iter_batches(pairs, batch_size_pairs):
            flat_ids = [str(text_id) for pair in batch for text_id in pair]
            batch_scores = detector.get_score(text=flat_ids)
            if batch_scores is None:
                raise RuntimeError(
                    f"Detector returned None for n_impostors={n_impostors}."
                )
            if len(batch_scores) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} scores for n_impostors={n_impostors}, "
                    f"got {len(batch_scores)}."
                )
            computed_scores.update(
                {
                    pair: float(score)
                    for pair, score in zip(batch, batch_scores)
                }
            )

    return computed_scores


def compute_prec_recall_f1_acc_dict_one_step_existing_mongodb(
    dataset_name: str,
    imp_gen_techniques: List[str],
    *,
    n_impostors: int = 50,
    n_potential_impostors: Optional[int] = None,
    min_available_impostors: int = 2,
    rounds: int = 100,
    top_n: int = 100000,
    portion_delete: float = 0.5,
    batch_size_pairs: int = 64,
    include_baselines: bool = True,
    upsample: bool = True,
) -> Dict[str, pd.DataFrame]:
    if imp_gen_techniques != [ONE_STEP_LLM_TECHNIQUE]:
        raise ValueError(
            f"This Mongo-only runner supports only {[ONE_STEP_LLM_TECHNIQUE]}, "
            f"got {imp_gen_techniques}."
        )
    if min_available_impostors < 2:
        raise ValueError(
            "min_available_impostors must be >= 2 because ImpostorDetector "
            "requires at least two impostors per text."
        )

    mongo_db = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    existing_scores = _load_existing_one_step_scores(
        mongo_db=mongo_db,
        dataset_name=dataset_name,
        rounds=rounds,
        n_impostors=None,
        n_potential_impostors=n_potential_impostors,
    )
    pair_docs = _load_pair_docs(mongo_db=mongo_db, dataset_name=dataset_name)

    unique_text_ids = {
        _to_object_id(pair["left_id"])
        for pair in pair_docs
    }.union({_to_object_id(pair["right_id"]) for pair in pair_docs})
    naive_counts = _count_naive_paraphrases(
        mongo_db=mongo_db,
        text_ids=unique_text_ids,
    )

    ordered_pairs: List[Tuple[ObjectId, ObjectId]] = []
    ground_truth: List[int] = []
    grouped_missing_pairs: Dict[int, List[Tuple[ObjectId, ObjectId]]] = defaultdict(list)
    skipped_without_score_or_paraphrases = 0

    for pair_doc in pair_docs:
        pair = (
            _to_object_id(pair_doc["left_id"]),
            _to_object_id(pair_doc["right_id"]),
        )
        if pair in existing_scores:
            ordered_pairs.append(pair)
            ground_truth.append(int(pair_doc["same"]))
            continue

        left_count = naive_counts.get(pair[0], 0)
        right_count = naive_counts.get(pair[1], 0)
        available_for_pair = min(left_count, right_count, n_impostors)
        if available_for_pair < min_available_impostors:
            skipped_without_score_or_paraphrases += 1
            continue

        ordered_pairs.append(pair)
        ground_truth.append(int(pair_doc["same"]))
        grouped_missing_pairs[available_for_pair].append(pair)

    logger.info(
        "Dataset '%s': %d total pairs, %d eligible, %d already scored, "
        "%d need scoring from existing naive paraphrases, %d skipped.",
        dataset_name,
        len(pair_docs),
        len(ordered_pairs),
        sum(1 for pair in ordered_pairs if pair in existing_scores),
        sum(len(pairs) for pairs in grouped_missing_pairs.values()),
        skipped_without_score_or_paraphrases,
    )
    if not ordered_pairs:
        raise NoEligiblePairsError(f"No eligible one_step_llm scores for dataset '{dataset_name}'.")

    computed_scores = _score_missing_pairs_from_existing_naive_paraphrases(
        grouped_pairs=grouped_missing_pairs,
        dataset_name=dataset_name,
        batch_size_pairs=batch_size_pairs,
        rounds=rounds,
        top_n=top_n,
        portion_delete=portion_delete,
        upsample=upsample,
    )
    scores_by_pair = {**existing_scores, **computed_scores}

    ordered_pairs = [pair for pair in ordered_pairs if pair in scores_by_pair]
    ground_truth = [
        int(pair_doc["same"])
        for pair_doc in pair_docs
        if (
            _to_object_id(pair_doc["left_id"]),
            _to_object_id(pair_doc["right_id"]),
        )
        in ordered_pairs
    ]
    scores = [scores_by_pair[pair] for pair in ordered_pairs]

    if len(ground_truth) != len(scores):
        raise RuntimeError(
            f"Ground truth and scores do not align: {len(ground_truth)} != {len(scores)}."
        )
    if not scores:
        raise RuntimeError(f"No eligible one_step_llm scores for dataset '{dataset_name}'.")

    results: Dict[str, pd.DataFrame] = {
        ONE_STEP_LLM_TECHNIQUE: compute_metrics_for_thresholds(
            ground_truth=ground_truth,
            scores=scores,
            thresholds=CONFIG.THRESHOLDS,
        )
    }

    if include_baselines:
        baselines = _build_baselines(dataset_name=dataset_name)
        text_test_pairs = mongo_db.get_texts_for_ids(
            text_ids=[text_id for pair in ordered_pairs for text_id in pair]
        )
        text_pairs = list(zip(text_test_pairs[0::2], text_test_pairs[1::2]))
        baseline_predictions = _score_baselines_for_pair_batches(
            baselines=baselines,
            pair_batches=[text_pairs],
            pair_id_batches=[ordered_pairs],
            dataset_name=dataset_name,
            mongoDB=mongo_db,
        )
        for baseline_name, baseline_scores in baseline_predictions.items():
            if len(baseline_scores) != len(ground_truth):
                raise RuntimeError(
                    f"Baseline {baseline_name}: expected {len(ground_truth)} scores, "
                    f"got {len(baseline_scores)}."
                )
            results[baseline_name] = compute_metrics_for_thresholds(
                ground_truth=ground_truth,
                scores=baseline_scores,
                thresholds=CONFIG.THRESHOLDS,
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute precision-recall curves for one_step_llm using only scores "
            "or one-step paraphrases already present in MongoDB."
        )
    )
    parser.add_argument(
        "--dataset_name",
        choices=[CONFIG.STUDENT_ESSAYS, CONFIG.BLOG, "all"],
        default="all",
    )
    parser.add_argument("--n_impostors", type=int, default=50)
    parser.add_argument("--n_potential_impostors", type=int, default=None)
    parser.add_argument("--min_available_impostors", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--top_n", type=int, default=100000)
    parser.add_argument("--portion_delete", type=float, default=0.5)
    parser.add_argument("--batch_size_pairs", type=int, default=64)
    parser.add_argument("--include_baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upsample", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Arguments: %s", args)

    compute_score_fn = partial(
        compute_prec_recall_f1_acc_dict_one_step_existing_mongodb,
        n_impostors=args.n_impostors,
        n_potential_impostors=args.n_potential_impostors,
        min_available_impostors=args.min_available_impostors,
        rounds=args.rounds,
        top_n=args.top_n,
        portion_delete=args.portion_delete,
        batch_size_pairs=args.batch_size_pairs,
        include_baselines=args.include_baselines,
        upsample=args.upsample,
    )


    dataset_names = (
        [CONFIG.STUDENT_ESSAYS, CONFIG.BLOG]
        if args.dataset_name == "all"
        else [args.dataset_name]
    )
    for dataset_name in dataset_names:
        try:
            run_prec_recall_curves(dataset_name=dataset_name, imp_gen_techniques=[ONE_STEP_LLM_TECHNIQUE],
                    compute_score_fn=compute_score_fn, )
        except NoEligiblePairsError as e:
            logger.warning("%s Skipping dataset.", e)


if __name__ == "__main__":
    main()
