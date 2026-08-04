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

N_DOCUMENTS = 200
BATCH_ID =
#"batch_6a7181d4d7ec8190a995ebeff68712b7"
#"batch_6a6e14edeab88190877185ef055c6abd"
#"batch_6a6cbb2a9da081908899ba4b7a797d85"
#"batch_6a6c6a14b4048190be47fb05fd558616"
    # "batch_6a6c698375948190a7b56af8232a69e0" # other model
#"batch_6a6c52f2d0948190887566be886a3c2d"
#"batch_6a6c39afca7c8190be2cc1a9a380939a"
#"batch_6a6b7334a24481908b2f8c9c22e099d0"
#"batch_6a6b50d2b3b48190af33eac552cd68fa"
#"batch_6a6b317454188190b304b53370e5a364"
#"batch_6a6b1aa3d88c8190a2a89496d3b959bf"
#"batch_6a6ae324decc8190a15285b01c0f835b"
#"batch_6a69f9469dac81909c9b7d43dcca7627"   # 100 docs with 50 paraphrases each
    # "batch_6a69dc56700c8190b88c1817c25f2f6a"
           #"batch_6a69d38466f88190983e275fef950de4"
    #"batch_6a69b0a71c008190a9f74327908d87af"
#"batch_6a68c7bf86108190958f69fdb6eeadc6"


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

    if len(text_ids) < n_documents:
        raise ValueError(
            f"Expected {n_documents} original texts without naive paraphrases "
            f"for query {query} and llm={llm!r}, found {len(text_ids)}."
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
        dataset_name=None,
        n_documents=N_DOCUMENTS,
    )
    logger.info("About to submit one batch for text_ids=%s", text_ids)

    paraphraser = BatchParaphraser()
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

    paraphraser = BatchParaphraser()
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

    submitted_batch_id = submit_one()
    collect_batch(batch_id=submitted_batch_id)
    #
    # To only collect an existing job, comment out submit_one above and call:
    # collect_batch(batch_id=BATCH_ID)


if __name__ == "__main__":
    main()
