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
"""Submit and/or collect one OpenAI Batch API paraphrase job.

This script intentionally keeps submit and collect as separate function calls in
``main`` so one of them can be commented out during manual testing.
"""

import argparse
import logging
import os
from typing import Optional

from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.batch_paraphraser import BatchParaphraser

logger = logging.getLogger(__name__)


def get_one_original_text_id(
    mongo: ParaphraseMongoDB,
    dataset_name: Optional[str],
) -> ObjectId:
    query = {}
    if dataset_name:
        query["dataset_name"] = dataset_name
    doc = mongo.original_collection.find_one(query, {"_id": 1}, sort=[("_id", 1)])
    if not doc:
        raise ValueError(f"No original text found for query: {query}")
    text_id = ObjectId(doc["_id"])
    logger.info("Selected original text_id=%s", text_id)
    return text_id


def submit_one() -> str:
    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    text_id = get_one_original_text_id(mongo=mongo)

    paraphraser = BatchParaphraser()
    paraphraser.add_text(
        text_id=text_id,
        metadata={"script": os.path.basename(__file__)},
    )
    batch_id = paraphraser.submit_batch()
    logger.info("Submitted batch_id=%s for text_id=%s", batch_id, text_id)
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
    # collect_batch(args, batch_id=args.batch_id)



if __name__ == "__main__":
    main()
