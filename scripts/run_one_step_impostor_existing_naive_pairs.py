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

import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

ONE_STEP_LLM_TECHNIQUE = "one_step_llm"


def _to_object_id(value) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)


def _count_naive_paraphrases(
    mongo_db: ParaphraseMongoDB, text_ids: List[ObjectId]
) -> Dict[ObjectId, int]:
    if not text_ids:
        return {}

    pipeline = [
        {"$match": {"text_id": {"$in": text_ids}}},
        {"$group": {"_id": "$text_id", "count": {"$sum": 1}}},
    ]
    return {
        _to_object_id(doc["_id"]): int(doc["count"])
        for doc in mongo_db.naive_paraphrase_collection.aggregate(
            pipeline, allowDiskUse=True
        )
    }


def _load_all_pairs_with_existing_naive_paraphrases_and_n_impostors(
    dataset_name: str,
    min_available_impostors: int = 1,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Load all pairs for a dataset, but keep only those where both texts have
    naive paraphrases in MongoDB.

    Returns
    -------
    text_id_pairs:
        Flattened list of pair IDs: [left_0, right_0, left_1, right_1, ...]
    ground_truth:
        List of labels aligned with pairs.
    n_impostors_per_pair:
        Per-pair n_impostors, defined as max(#naive_left, #naive_right).
    """
    if min_available_impostors < 1:
        raise ValueError(
            f"min_available_impostors must be >= 1, got {min_available_impostors}"
        )

    mongo_db = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    pair_docs = list(
        mongo_db.all_pairs_collection.find(
            {"dataset_name": dataset_name},
            {"left_id": 1, "right_id": 1, "same": 1},
        ).sort("_id", 1)
    )

    if not pair_docs:
        logger.warning("No pairs found for dataset '%s'.", dataset_name)
        return [], [], []

    unique_text_ids = list(
        {
            _to_object_id(pair["left_id"])
            for pair in pair_docs
        }.union({_to_object_id(pair["right_id"]) for pair in pair_docs})
    )
    naive_counts = _count_naive_paraphrases(mongo_db=mongo_db, text_ids=unique_text_ids)

    text_id_pairs: List[str] = []
    ground_truth: List[int] = []
    n_impostors_per_pair: List[int] = []

    skipped_without_naive = 0
    skipped_under_min_impostors = 0
    for pair in pair_docs:
        left_id = _to_object_id(pair["left_id"])
        right_id = _to_object_id(pair["right_id"])
        left_count = naive_counts.get(left_id, 0)
        right_count = naive_counts.get(right_id, 0)

        if left_count == 0 and right_count == 0:
            skipped_without_naive += 1
            continue

        n_impostors = max(left_count, right_count)
        if n_impostors < min_available_impostors:
            skipped_under_min_impostors += 1
            continue

        text_id_pairs.extend([str(left_id), str(right_id)])
        ground_truth.append(int(pair["same"]))
        n_impostors_per_pair.append(n_impostors)

    logger.info(
        "Dataset '%s': %d total pairs, %d kept, %d skipped (missing naive), %d skipped (<%d impostors).",
        dataset_name,
        len(pair_docs),
        len(ground_truth),
        skipped_without_naive,
        skipped_under_min_impostors,
        min_available_impostors,
    )
    return text_id_pairs, ground_truth, n_impostors_per_pair


def load_all_pairs_with_existing_naive_paraphrases(
    dataset_name: str,
    min_available_impostors: int = 1,
) -> Tuple[List[str], List[int]]:
    text_id_pairs, ground_truth, _ = (
        _load_all_pairs_with_existing_naive_paraphrases_and_n_impostors(
            dataset_name=dataset_name,
            min_available_impostors=min_available_impostors,
        )
    )
    return text_id_pairs, ground_truth


def run_one_step_impostor_on_existing_naive_pairs(
    dataset_name: str,
    batch_size_pairs: int = 64,
    min_available_impostors: int = 1,
) -> List[float]:
    text_id_pairs, ground_truth, n_impostors_per_pair = (
        _load_all_pairs_with_existing_naive_paraphrases_and_n_impostors(
            dataset_name=dataset_name,
            min_available_impostors=min_available_impostors,
        )
    )

    pair_count = len(ground_truth)
    if pair_count == 0:
        logger.warning("No eligible pairs to score for dataset '%s'.", dataset_name)
        return []

    grouped_pair_indices: Dict[int, List[int]] = defaultdict(list)
    for pair_idx, n_impostors in enumerate(n_impostors_per_pair):
        grouped_pair_indices[n_impostors].append(pair_idx)

    scores: List[float] = [0.0] * pair_count
    detectors: Dict[int, ImpostorDetector] = {}

    for n_impostors, pair_indices in sorted(grouped_pair_indices.items()):
        detector = detectors.get(n_impostors)
        if detector is None:
            detector = ImpostorDetector(
                impostor_technique=ONE_STEP_LLM_TECHNIQUE,
                n_impostors=max(15,n_impostors),
                dataset_name=dataset_name,
            )
            detectors[n_impostors] = detector

        logger.info(
            "Scoring %d pairs with technique='%s' and n_impostors=%d.",
            len(pair_indices),
            ONE_STEP_LLM_TECHNIQUE,
            n_impostors,
        )
        for start in range(0, len(pair_indices), batch_size_pairs):
            batch_pair_indices = pair_indices[start : start + batch_size_pairs]
            batch_text_ids: List[str] = []
            for pair_idx in batch_pair_indices:
                pair_offset = pair_idx * 2
                batch_text_ids.extend(
                    [text_id_pairs[pair_offset], text_id_pairs[pair_offset + 1]]
                )

            batch_scores = detector.get_score(text=batch_text_ids)
            if batch_scores is None:
                raise RuntimeError(
                    f"Detector returned None for n_impostors={n_impostors}, "
                    f"batch starting at index {start}."
                )
            if len(batch_scores) != len(batch_pair_indices):
                raise RuntimeError(
                    f"Expected {len(batch_pair_indices)} scores for n_impostors={n_impostors}, "
                    f"got {len(batch_scores)}."
                )

            for local_idx, pair_idx in enumerate(batch_pair_indices):
                scores[pair_idx] = float(batch_scores[local_idx])

    logger.info("Finished scoring %d pairs for dataset '%s'.", pair_count, dataset_name)
    return scores


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    scores = run_one_step_impostor_on_existing_naive_pairs(
        dataset_name=CONFIG.STUDENT_ESSAYS,
        batch_size_pairs=64,
        min_available_impostors=5,
    )
    logger.info("Computed %d scores.", len(scores))


if __name__ == "__main__":
    main()
