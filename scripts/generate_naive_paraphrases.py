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

# Generate naive paraphrases for documents in MongoDB.
import argparse
import logging
import os
from typing import Iterable, Optional

from bson import ObjectId
from openai import OpenAI

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.naive_impostor_generator import (
    NaiveImpostorGenerator,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.one_step_paraphrasers import SAIAParaphraser

logger = logging.getLogger(__name__)


def resolve_saia_api_key() -> Optional[str]:
    for key in CONFIG.SAIA_KEYS:
        if key and key.strip():
            return key.strip()
    return None


def validate_saia_api_key(api_key: str, model_id: str = CONFIG.SAIA_MODEL) -> None:
    client = OpenAI(
        base_url=CONFIG.SAIA_URL if os.path.exists("/Users/klara") else os.environ["SAIA_URL"],
        api_key=os.environ["SAIA_KEY"],
    )
    client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Reply with OK."}],
        max_tokens=4,
        temperature=0,
    )


def iter_original_text_ids(
    mongo: ParaphraseMongoDB,
    dataset_name: Optional[str],
    limit: Optional[int],
) -> Iterable[ObjectId]:
    query = {}
    if dataset_name:
        query["dataset_name"] = dataset_name
    cursor = mongo.original_collection.find(query, {"_id": 1}).sort("_id", 1)
    count = 0
    for doc in cursor:
        yield ObjectId(doc["_id"])
        count += 1
        if limit is not None and count >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate naive paraphrases for MongoDB original texts."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=CONFIG.STUDENT_ESSAYS,
        help="Filter by dataset name stored in MongoDB (optional).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of texts to process (optional).",
    )
    parser.add_argument(
        "--n-impostors",
        type=int,
        default=10,
        help="Number of naive paraphrases to generate per text.",
    )
    parser.add_argument(
        "--top-n-freq-words",
        type=int,
        default=100000,
        help="Top-N frequent words for similarity selection (passed through).",
    )
    parser.add_argument(
        "--skip-existing",
        default=True,
        action="store_true",
        help="Skip texts that already have >= n-impostors paraphrases.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    api_key = resolve_saia_api_key()
    if not api_key:
        logger.error(
            "No SAIA API key configured. Set SAIA_KEY (or SAIA_KEY_K) in your environment."
        )
        raise SystemExit(1)
    try:
        validate_saia_api_key(api_key=api_key)
        logger.info("SAIA key preflight check passed.")
    except Exception as e:
        logger.error("SAIA key preflight check failed: %s", e)
        raise SystemExit(1)

    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    logger.info("Running in sequential mode (no parallel workers).")

    def _init_saia_clients(
        generator: NaiveImpostorGenerator, api_key: Optional[str]
    ) -> None:
        for paraphraser in generator.paraphrasers:
            if isinstance(paraphraser, SAIAParaphraser):
                if api_key and api_key.strip():
                    paraphraser._init_client(saia_api_key=api_key)
                else:
                    # Keep existing working client; do not overwrite with empty key
                    logger.warning(
                        "No valid SAIA key configured; keeping existing client."
                    )
                assert (
                    paraphraser.client.api_key is not None
                ), "SAIA client api_key is None after initialization."

    generator = NaiveImpostorGenerator(
        n_impostors=args.n_impostors, top_n_freq_words=args.top_n_freq_words
    )
    _init_saia_clients(generator, api_key=api_key)
    generator._set_dataset_name(dataset_name=args.dataset)

    processed = 0
    for text_id in iter_original_text_ids(mongo, args.dataset, args.limit):
        if args.skip_existing:
            existing_cursor = generator.mongoDB.find_document_by_multiple_fields(
                collection=generator.mongoDB.naive_paraphrase_collection,
                search_args={"text_id": text_id},
            )
            existing = list(existing_cursor)
            if len(existing) >= args.n_impostors:
                logger.info(
                    "Skipping text_id=%s (already %d paraphrases).",
                    text_id,
                    len(existing),
                )
                continue
        try:
            generator.generate_impostors(text=None, text_id=text_id)
            processed += 1
        except Exception as e:
            logger.warning(
                "Failed to generate for text_id=%s: %s",
                text_id,
                e,
            )

    logger.info("Done. Processed %d text(s).", processed)


if __name__ == "__main__":
    main()
