# Copyright 2025
#
# Generate naive paraphrases for documents in MongoDB.
import argparse
import logging
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional

from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.naive_impostor_generator import (
    NaiveImpostorGenerator,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.one_step_paraphrasers import SAIAParaphraser

logger = logging.getLogger(__name__)


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
        default=True,
        action="store_true",
        help="Skip texts that already have >= n-impostors paraphrases.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,#None,
        help="Number of parallel workers (defaults to number of SAIA keys).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    worker_count = len(CONFIG.SAIA_KEYS)
    if args.workers is not None:
        worker_count = max(1, min(args.workers, worker_count))
    logger.info(f"Worker count: {worker_count}")

    task_queue: queue.Queue[Optional[ObjectId]] = queue.Queue()

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
                        "No valid SAIA key for worker; keeping existing client."
                    )
                assert (
                    paraphraser.client.api_key is not None
                ), "SAIA client api_key is None after initialization."

    def _worker(worker_idx: int) -> int:
        generator = NaiveImpostorGenerator(
            n_impostors=args.n_impostors, top_n_freq_words=args.top_n_freq_words
        )
        _init_saia_clients(generator, api_key=CONFIG.SAIA_KEYS[worker_idx])
        processed = 0
        while True:
            text_id = task_queue.get()
            if text_id is None:
                task_queue.task_done()
                break
            if args.skip_existing:
                existing_cursor = generator.mongoDB.find_document_by_multiple_fields(
                    collection=generator.mongoDB.naive_paraphrase_collection,
                    search_args={"text_id": text_id},
                )
                existing = list(existing_cursor)
                if len(existing) >= args.n_impostors:
                    logger.info(
                        "Worker %d skipping text_id=%s (already %d paraphrases).",
                        worker_idx,
                        text_id,
                        len(existing),
                    )
                    task_queue.task_done()
                    continue
            try:
                generator.generate_impostors(text=None, text_id=text_id)
                processed += 1
            except Exception as e:
                logger.warning(
                    "Worker %d failed to generate for text_id=%s: %s",
                    worker_idx,
                    text_id,
                    e,
                )
            task_queue.task_done()
        return processed

    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_worker, idx)
                for idx in range(worker_count)
            ]
            for text_id in iter_original_text_ids(mongo, args.dataset, args.limit):
                task_queue.put(text_id)
            for _ in range(worker_count):
                task_queue.put(None)
            processed = sum(f.result() for f in futures)
    else:
        for text_id in iter_original_text_ids(mongo, args.dataset, args.limit):
            task_queue.put(text_id)
            worker = _worker(0)

    logger.info("Done. Processed %d text(s).", processed)


if __name__ == "__main__":
    main()
