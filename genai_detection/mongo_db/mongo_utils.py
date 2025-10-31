import os
from pathlib import Path
from genai_detection.config import CONFIG
from pymongo import MongoClient
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class ParaphraseMongoDB:

    def __init__(self):
        uri = f"mongodb://{CONFIG.MONGO_USER}:{CONFIG.MONGO_PASSWORD}@{CONFIG.MONGO_HOST}/"
        self.client = MongoClient(uri)
        try:
            self.client.admin.command("ping")
            print("Successfully connected as MongoDB root user!")
        except Exception as e:
            print("Connection failed:", e)
            raise e

        self.db = self.client[CONFIG.MONGO_DATABASE or "impostors"]
        self.original_collection = self.db[CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION or "original_text"]

        # Create collections if they don't exist
        for collection_name in [CONFIG.MONGO_PARAPHRASE_COLLECTION, CONFIG.MONGO_SCORE_COLLECTION]:
            if collection_name not in self.db.list_collection_names():
                self.db.create_collection(collection_name)
                print(f"Created collection: {collection_name}")
            else:
                print(
                    f"Collection '{collection_name}' already exists. Skipping creation."
                )

        self.paraphrase_collection = self.db[CONFIG.MONGO_PARAPHRASE_COLLECTION]
        self.score_collection = self.db[CONFIG.MONGO_SCORE_COLLECTION]

    def update_document(self, collection, text_id: str, update_data: dict) -> int:
        """
        Update an entry in the specified collection by its text ID.
        If no document with text_id exists, no action is taken.
        If update_data contains fields that already exist, they will be overwritten, if fields are non-existent, they will be created.
        :param collection: The MongoDB collection to update.
        :param text_id: The ID of the text document to update.
        :param update_data: A dictionary containing the fields to update.
        :return: The number of documents modified (0 or 1).
        """
        assert isinstance(update_data, dict), "update_data must be a dictionary"
        assert ("text_id" not in update_data) or (text_id == update_data["text_id"]), "update_data must not contain 'text_id' key or have the same value as text_id parameter"
        result = collection.update_one(
            {"_id": text_id},
            {"$set": update_data}
        )
        return result.modified_count

    def find_document(self, collection, id: str) -> dict | None:
        """
        Find an entry in the specified collection by its text ID.
        :param collection: The MongoDB collection to search.
        :param text_id: The ID of the text document to find.
        :return: The found document as a dictionary, or None if not found.
        """
        document = collection.find_one({"_id": text_id})
        return document