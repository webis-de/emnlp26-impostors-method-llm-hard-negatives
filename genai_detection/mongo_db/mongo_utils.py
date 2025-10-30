import os
from pathlib import Path
from genai_detection.config import CONFIG
from pymongo import MongoClient
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class ParaphraseMongoDB:

    def __init__(self):
        uri = f"mongodb://{CONFIG.MONGO_USER}:{CONFIG.MONGO_PASSWORD}@{CONFIG.MONGO_HOST}/"
        client = MongoClient(uri)
        try:
            client.admin.command("ping")
            print("Successfully connected as MongoDB root user!")
        except Exception as e:
            print("Connection failed:", e)
            raise e

        self.db = client[CONFIG.MONGO_DATABASE or "impostors"]
        self.original_collection = self.db[CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION or "original_text"]
        self.paraphrase_collection = self.db[CONFIG.MONGO_PARAPHRASE_COLLECTION or "paraphrase"]
        score_collection_name = CONFIG.MONGO_SCORE_COLLECTION or "score"


        # Create score collection if it doesn't exist
        if score_collection_name not in self.db.list_collection_names():
            self.db.create_collection(score_collection_name)
            print(f"Created collection: {score_collection_name}")
        else:
            print(f"Collection '{score_collection_name}' already exists. Skipping creation.")

        self.score_collection = self.db[score_collection_name]