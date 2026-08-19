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
"""Submit and/or collect one OpenAI Batch API paraphrase job for two documents.

This script intentionally keeps submit and collect as separate function calls in
``main`` so one of them can be commented out during manual testing.
"""

import logging
import os
from typing import List, Optional

from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.batch_paraphraser import BatchParaphraser

logger = logging.getLogger(__name__)

N_DOCUMENTS = 400
BATCH_ID = "batch_6a855a4cc9f08190a36886c00e04bdca" # blogs, # TODO:
DATASET_NAME = CONFIG.BLOG #CONFIG.STUDENT_ESSAYS  # None
# LLM = CONFIG.BATCH_OPENAI_MODEL_LUNA if DATASET_NAME==CONFIG.BLOG else CONFIG.BATCH_OPENAI_MODEL
LLM = CONFIG.BATCH_OPENAI_MODEL


def get_original_text_ids_without_naive_paraphrases(
    mongo: ParaphraseMongoDB,
    dataset_name: Optional[str],
    n_documents: int,
    llm: str = CONFIG.BATCH_OPENAI_MODEL,
) -> List[ObjectId]:
    query = {}
    if dataset_name:
        query["dataset_name"] = dataset_name

    text_ids = []
    seen = set()
    missing_pair_ids = []
    cursor = mongo.all_pairs_collection.find(
        query,
        {"left_id": 1, "right_id": 1},
    ).sort("_id", 1)
    for pair in cursor:
        pair_text_ids = [ObjectId(pair["left_id"]), ObjectId(pair["right_id"])]
        existing_ids = {
            ObjectId(doc["text_id"])
            for doc in mongo.naive_paraphrase_collection.find(
                {
                    "text_id": {"$in": pair_text_ids},
                    "llm": llm,
                },
                {"text_id": 1},
            )
        }
        missing_ids = [text_id for text_id in pair_text_ids if text_id not in existing_ids]
        if len(missing_ids) == 1:
            text_id = missing_ids[0]
            if text_id not in seen:
                text_ids.append(text_id)
                seen.add(text_id)
        elif len(missing_ids) == 2:
            missing_pair_ids.extend(missing_ids)
        if len(text_ids) >= n_documents:
            break

    for text_id in missing_pair_ids:
        if len(text_ids) >= n_documents:
            break
        if text_id in seen:
            continue
        text_ids.append(text_id)
        seen.add(text_id)

    if len(text_ids) < n_documents:
        cursor = mongo.original_collection.find(query, {"_id": 1}).sort("_id", 1)
        for doc in cursor:
            text_id = ObjectId(doc["_id"])
            if text_id in seen:
                continue
            existing = mongo.naive_paraphrase_collection.find_one(
                {
                    "text_id": text_id,
                    "llm": llm,
                },
                {"_id": 1},
            )
            if existing:
                continue
            text_ids.append(text_id)
            seen.add(text_id)
            if len(text_ids) >= n_documents:
                break

    if not text_ids:
        raise ValueError(
            "Expected at least one original text without naive paraphrases "
            f"for query {query} and llm={llm!r}, found 0."
        )
    if len(text_ids) < n_documents:
        logger.info(
            "Only found %d original texts without naive paraphrases for query %s "
            "and llm=%r; submitting a smaller batch than requested (%d).",
            len(text_ids),
            query,
            llm,
            n_documents,
        )
    logger.info(
        "Selected original text_ids without naive paraphrases for llm=%s: %s",
        llm,
        text_ids,
    )
    return text_ids


def submit_one() -> Optional[str]:
    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

    text_ids = get_original_text_ids_without_naive_paraphrases(
        mongo=mongo,
        dataset_name=DATASET_NAME,
        n_documents=N_DOCUMENTS,
        llm=LLM,
    )
    logger.info(
        "About to submit one batch for %s text_ids=%s for dataset %s using %s model (batched)",
        len(text_ids),
        text_ids,
        DATASET_NAME,
        LLM,
    )

    paraphraser = BatchParaphraser(model_id=LLM)
    custom_ids = paraphraser.add_texts(
        text_ids=text_ids,
        metadata={"script": os.path.basename(__file__)},
    )
    logger.info("Queued %d batch requests.", len(custom_ids))
    batch_id = paraphraser.submit_batch()
    logger.info("Submitted batch_id=%s for text_ids=%s", batch_id, text_ids)
    return batch_id


def collect_batch(batch_id: Optional[str]) -> None:
    if not batch_id:
        logger.info("No batch_id provided; skipping collection.")
        return

    paraphraser = BatchParaphraser(model_id=LLM)
    summary = paraphraser.collect_batch(batch_id=batch_id, save=True)
    logger.info("Collect summary: %s", summary)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # from openai import OpenAI
    #
    # client = OpenAI(base_url=CONFIG.OPENAI_URL, api_key=CONFIG.OPENAI_KEY, )
    #
    # batches = client.batches.list(limit=10)
    #
    # for batch in batches.data:
    #     print(batch.id, batch.status, batch.created_at, batch.request_counts)

    # submitted_batch_id = submit_one()
    # collect_batch(batch_id=submitted_batch_id)
    #
    # To only collect an existing job, comment out submit_one above and call:
    collect_batch(batch_id=BATCH_ID)


if __name__ == "__main__":
    main()
