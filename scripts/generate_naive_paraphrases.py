# Copyright 2025
#
# Generate naive paraphrases for documents in MongoDB.
import argparse
import logging
import os
from typing import Iterable, Optional

from bson import ObjectId

from genai_detection.impostor_generators.naive_impostor_generator import (
    NaiveImpostorGenerator,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)


def iter_original_text_ids(
    mongo: ParaphraseMongoDB,
    dataset_name: Optional[str],
    limit: Optional[int],
) -> Iterable[ObjectId]:
    query = {}
    if dataset_name:
        query["dataset"] = dataset_name
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
        default=None,
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
        action="store_true",
        help="Skip texts that already have >= n-impostors paraphrases.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    generator = NaiveImpostorGenerator(
        n_impostors=args.n_impostors, top_n_freq_words=args.top_n_freq_words
    )

    processed = 0
    for text_id in iter_original_text_ids(mongo, args.dataset, args.limit):
        if args.skip_existing:
            existing_cursor = mongo.find_document_by_multiple_fields(
                collection=mongo.naive_paraphrase_collection, search_args={"text_id": text_id}
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
            logger.warning("Failed to generate for text_id=%s: %s", text_id, e)

    logger.info("Done. Processed %d text(s).", processed)


if __name__ == "__main__":
    main()
