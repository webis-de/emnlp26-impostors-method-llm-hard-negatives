import hashlib
import logging

from pymongo import errors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", )
import os
from pathlib import Path

from genai_detection.config import CONFIG
from genai_detection.dataset.student_essays_dataset_loader import StudentEssayDatasetLoader
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    base_dir = (
        Path(__file__).resolve().parents[2]
        / CONFIG.DATA_BASE_PATH
        / "student_essays/Intro2006"
    )
    assert (
        base_dir.exists()
    ), f"Path {base_dir} to student essays dataset does not exist."

    logger.info("Starting initialization: loading original_text dataset...")

    # Connect to MongoDB (default host/port for container)
    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    collection_name = CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION

    # Create unique index on text_hash
    new_collection = mongoDB.reset_collection(collection_name=collection_name)
    new_collection.create_index("text_hash", unique=True)

    # Dataset base path from environment
    dataset_base_path = (PROJECT_ROOT / CONFIG.PATH2STUDENT_ESSAYS).parent
    assert dataset_base_path.exists(), f"Dataset base path does not exist: {dataset_base_path}"

    dataset_name = CONFIG.STUDENT_ESSAYS
    logger.info(f"{dataset_name} dataset base path: {dataset_base_path}")

    loader = StudentEssayDatasetLoader(path=base_dir)
    original_texts_df = loader.load_texts()

    # Compute hashes
    original_texts_df["text_hash"] = original_texts_df["text"].apply(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
    )
    logger.info(f"{len(original_texts_df)} original texts found")

    # Convert to list of dicts
    records = original_texts_df.to_dict(orient="records")

    # Fetch existing hashes in bulk to avoid repeated DB queries
    existing_hashes = set(mongoDB.original_collection.distinct("text_hash"))

    # Filter new docs
    docs_to_insert = [
        {
            "author": r.pop("author_id"),
            "assignment": r.pop("task"),
            **r,
            "dataset": dataset_name,
        }
        for r in records
        if r["text_hash"] not in existing_hashes
    ]

    # Insert documents
    if docs_to_insert:
        try:
            result = mongoDB.original_collection.insert_many(
                docs_to_insert, ordered=False
            )
            logger.info(
                f"Inserted {len(result.inserted_ids)} documents into '{collection_name}'"
            )
        except errors.BulkWriteError:
            logger.info(
                "Duplicate key error encountered during insertMany. Some documents may already exist."
            )
    else:
        logger.info("No new documents to insert.")

    logger.info("Initialization complete.")

if __name__ == "__main__":
    main()
