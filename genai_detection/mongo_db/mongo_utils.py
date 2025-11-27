import logging
from pathlib import Path
from typing import Optional

from bson import ObjectId
from pymongo import MongoClient

from genai_detection.config import CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

class ParaphraseMongoDB:

    def __init__(self):
        uri = f"mongodb://{CONFIG.MONGO_USER}:{CONFIG.MONGO_PASSWORD}@{CONFIG.MONGO_HOST}/"
        self.client = MongoClient(uri)
        try:
            self.client.admin.command("ping")
            logging.info("Successfully connected as MongoDB root user!")
        except Exception as e:
            logging.warning("Connection failed: %s", e)
            raise e

        self.db = self.client[CONFIG.MONGO_DATABASE or "impostors"]
        self.original_collection = self.db[
            CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION or "original_text"
        ]

        # Create collections if they don't exist
        for collection_name in [
            CONFIG.MONGO_PARAPHRASE_COLLECTION,
            CONFIG.MONGO_PARAPHRASE_SCORE_COLLECTION,
            CONFIG.MONGO_IMPOSTOR_OUTPUT_COLLECTION,
            CONFIG.MONGO_ON_THE_FLY_COLLECTION,
        ]:
            if collection_name not in self.db.list_collection_names():
                self.db.create_collection(collection_name)
                logging.info(f"Created collection: {collection_name}")
            else:
                logging.info(
                    f"Collection '{collection_name}' already exists. Skipping creation."
                )

        self.paraphrase_collection = self.db[CONFIG.MONGO_PARAPHRASE_COLLECTION]    # llm paraphrases
        self.on_the_fly_collection = self.db[CONFIG.MONGO_ON_THE_FLY_COLLECTION] # on-the-fly impostors
        self.paraphrase_score_collection = self.db[CONFIG.MONGO_PARAPHRASE_SCORE_COLLECTION]  # paraphrase scores
        self.impostor_output_collection = self.db[CONFIG.MONGO_IMPOSTOR_OUTPUT_COLLECTION]  # output of impostor
        # approach

    @staticmethod
    def update_document(collection, _id: str, update_data: dict) -> int:
        """
        Update an entry in the specified collection by its ID.
        If no document with _id exists, no action is taken.
        If update_data contains fields that already exist, they will be overwritten, if fields are non-existent, they will be created.
        :param collection: The MongoDB collection to update.
        :param text_id: The ID of the text document to update.
        :param update_data: A dictionary containing the fields to update.
        :return: The number of documents modified (0 or 1).
        """
        assert isinstance(update_data, dict), "update_data must be a dictionary"
        result = collection.update_one({"_id": _id}, {"$set": update_data})
        return result.modified_count

    @staticmethod
    def insert_document(collection, insert_data: dict):
        """
        Insert an entry in the specified collection.
        :param collection: The MongoDB collection to insert.
        :param insert_data: A dictionary with the data to insert.
        :return: -
        """
        assert isinstance(insert_data, dict), f"insert_data must be a dictionary, but is {type(insert_data)}."
        collection.insert_one(insert_data)

    @staticmethod
    def find_document_by_id(collection, document_id: str):
        """
        Find an entry in the specified collection by its text ID.
        :param collection: The MongoDB collection to search.
        :param document_id: The ID of the text document to find.
        :return: The found document as a cursor object, or None if not found.
        """
        document = collection.find({"_id": ObjectId(document_id)})
        return document

    @staticmethod
    def find_document_by_non_id_field(collection, document_field_name: str, document_value:str):
        """
        Find an entry in the specified collection by its value of a non-id field.
        :param collection: The MongoDB collection to search.
        :param document_value: The value of a non-id field.
        :param document_field_name: The name of the non-id field.
        :return: The found document as a cursor object, or None if not found.
        """
        document = collection.find({document_field_name: document_value})
        return document

    def get_text_or_id_from_orginal_collection(self, text: Optional[str], text_id: Optional[str]):
        """
        Returns text and text_id from a document in the original collection.
        :param text: input text to generate impostors for (i.e., the candidate text)
        :param text_id: ID in a mongo database containing texts to retrieve impostors from.
        :return: A tuple containing text and text_id (can be none if not yet in collection).
        """
        assert text or text_id, "Either text or text_id must be provided."
        # Get text from mongodb if missing
        if not text:
            # Should only contain one element because _id is the primary key
            text = self.find_document_by_id(
                collection=self.original_collection, document_id=text_id
            )[0]["text"]
            assert (
                (text is not None) and type(text) == str and len(text) > 0
            ), f"Text ID {text_id} not found."

        if not text_id:
            text_id = self.find_document_by_non_id_field(collection=self.original_collection, document_field_name="text", document_value=text)[0]["_id"]
        # Save text in mongoDB if not yet present
        if text and text_id:
            existing = self.find_document_by_id(
                collection=self.original_collection, document_id=text_id
            )
            if existing is None:
                raise Exception(f"Document ID {text_id} not found.")
                self.mongoDB.insert_document(
                    collection=self.original_collection,
                    insert_data={"text": text},
                )
        return text, text_id

    def find_paraphrases(self, document_id: str):
        """
        Find an entry in the specified collection by its text ID.
        :param document_id: The ID of the original document. Paraphrases have different _id.
        :return: The found document as a cursor, or None if not found.

        https://www.mongodb.com/docs/manual/reference/method/db.collection.find/ (06.11.2025)
        """
        documents = self.paraphrase_collection.find({"text_id": document_id})
        return documents
