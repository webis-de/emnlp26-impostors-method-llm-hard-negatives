import os
import hashlib
from pathlib import Path
from genai_detection.config import CONFIG
from pymongo import MongoClient, errors
PROJECT_ROOT = Path(__file__).resolve().parents[2]

print("Starting initialization: loading original_text dataset...")

# Get database name from environment or default
db_name = CONFIG.MONGO_DATABASE or "impostors"

# Connect to MongoDB (default host/port for container)
uri = f"mongodb://{CONFIG.MONGO_USER}:{CONFIG.MONGO_PASSWORD}@{CONFIG.MONGO_HOST}/"
client = MongoClient(uri)
db = client[db_name]

# Collection name
collection_name = CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION or "original_text"

# Create collection if it does not exist
if collection_name not in db.list_collection_names():
    db.create_collection(collection_name)
    print(f"Created collection: {collection_name}")
else:
    print(f"Collection '{collection_name}' already exists. Skipping creation.")

# Create unique index on text_hash
db[collection_name].create_index("text_hash", unique=True)

# Dataset base path from environment
dataset_base_path = (PROJECT_ROOT / CONFIG.PATH2STUDENT_ESSAYS).parent
assert dataset_base_path.exists(), f"Dataset base path does not exist: {dataset_base_path}"

dataset_name = CONFIG.STUDENT_ESSAYS
print(f"{dataset_name} dataset base path: {dataset_base_path}")

docs_to_insert = []

# List directories under dataset_base_path
for dir_name in os.listdir(dataset_base_path):
    dir_path = dataset_base_path / dir_name
    if not dir_path.is_dir() or not dir_name.startswith("Ass"):
        continue

    print(f"Processing assignment directory: {dir_name}")

    # List .txt files in this assignment directory
    for file_name in os.listdir(dir_path):
        if not file_name.endswith(".txt"):
            continue

        file_path = dir_path / file_name
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        except Exception as e:
            print(f"Could not read file: {file_path}. Error: {e}")
            continue

        author_name = file_name.replace(".txt", "")
        text_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()

        # Check if document already exists
        exists = db[collection_name].find_one(
            {"author": author_name, "assignment": dir_name, "text_hash": text_hash}
        )
        if exists:
            continue

        doc = {
            "text": text_content,
            "author": author_name,
            "dataset": dataset_name,
            "assignment": dir_name,
            "text_hash": text_hash,
        }

        docs_to_insert.append(doc)

# Insert documents
if docs_to_insert:
    try:
        result = db[collection_name].insert_many(docs_to_insert, ordered=False)
        print(f"Inserted {len(result.inserted_ids)} documents into '{collection_name}'")
    except errors.BulkWriteError as e:
        print(
            "Duplicate key error encountered during insertMany. Some documents may already exist."
        )
else:
    print("No documents found to insert.")

print("Initialization complete.")
