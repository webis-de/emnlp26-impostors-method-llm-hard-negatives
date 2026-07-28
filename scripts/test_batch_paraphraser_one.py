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

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.batch_paraphraser import BatchParaphraser

logger = logging.getLogger(__name__)


def get_original_text_ids(
    mongo: ParaphraseMongoDB,
    dataset_name: Optional[str],
    n_documents: int,
) -> List[ObjectId]:
    query = {}
    if dataset_name:
        query["dataset_name"] = dataset_name
    cursor = mongo.original_collection.find(query, {"_id": 1}).sort("_id", 1).limit(
        n_documents
    )
    text_ids = [ObjectId(doc["_id"]) for doc in cursor]
    if len(text_ids) < n_documents:
        raise ValueError(
            f"Expected {n_documents} original texts for query {query}, "
            f"found {len(text_ids)}."
        )
    logger.info("Selected original text_ids=%s", text_ids)
    return text_ids


def submit_one() -> str:
    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    text_ids = get_original_text_ids(mongo=mongo, dataset_name=None, n_documents=2)
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

    submitted_batch_id = submit_one()
    collect_batch(batch_id=submitted_batch_id)

    # To only collect an existing job, comment out submit_one above and call:
    # collect_batch(batch_id=BATCH_ID)


BATCH_ID = "batch_6a68c7bf86108190958f69fdb6eeadc6"
if __name__ == "__main__":
    main()
