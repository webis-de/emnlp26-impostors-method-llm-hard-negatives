# Copyright 2024 Klara M. Gutekunst, Webis
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

import random
import typing as t
from abc import ABC, abstractmethod
from itertools import combinations

import pandas as pd
from datasets import (
    DatasetDict, Features, Value,
)
from pymongo import errors

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.one_step_paraphrasers import *
from genai_detection.paraphrasing.two_step_paraphrasers import *
from genai_detection.util import preprocess_text as _preprocess_text

logger = logging.getLogger(__name__)

random.seed(42)
# Koppel et al. (2004): 500 words
# Bevendorff et al. (2019): 700 words (https://www.degruyterbrill.com/document/doi/10.1515/itit-2019-0046/html?casa_token=pbCaF7FgUXoAAAAA:8Vw71FUWE5spAbSsEuGGTdIjjm_o1_eb_inHwU3BR6eSrdVMOYy3--iqvDJwCV7EQ1HWtQBh610)
# Bevendorff et al. (2025): 3000 characters (https://aclanthology.org/2025.findings-acl.194.pdf)
MIN_NUM_WORDS = 700  # minimum number of words in a text to be considered valid

# Canonical column names used throughout the loader
AUTHOR_COL_NAME = "author"
ASSIGNMENT_COL_NAME = "assignment"

# === BASE CLASS ===


class BaseDatasetLoader(ABC):
    def __init__(self, name: str):
        self.name = name
        # Connect to MongoDB (default host/port for container)
        self.mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
        self.features = Features(
            {
                "_id": Value("string"),
                "left_id": Value("string"),
                "right_id": Value("string"),
                f"left_{AUTHOR_COL_NAME}": Value("string"),
                f"right_{AUTHOR_COL_NAME}": Value("string"),
                f"left_{ASSIGNMENT_COL_NAME}": Value("string"), # topic from blogs corpus is mapped to this
                f"right_{ASSIGNMENT_COL_NAME}": Value("string"),
                "dataset_name": Value("string"),
                "same": Value("bool"),
            }
        )

    @abstractmethod
    def load(self) -> DatasetDict:
        pass

    def preprocess(
        self, text: t.Union[str, t.Iterable[str]]
    ) -> t.Union[str, t.List[str]]:
        return _preprocess_text(text=text)

    @staticmethod
    def _load_jsonl(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _load_ids(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    def _generate_pairs(self, data):
        """
        Generate pairs of texts from the dataset irrespective of confounders (i.e. naive combinations).
        Dataset is expected to be a list of dictionaries with 'text' and 'author' keys.
        Each pair consists of two texts, their authors, and a boolean indicating if they are from the same author.
        """
        pairs = []
        for a, b in combinations(data, 2):
            pairs.append(
                {
                    "pair": [a["text"], b["text"]],
                    "authors": [a["author"], b["author"]],
                    "same": a["author"] == b["author"],
                }
            )

        return pairs

    def save2mongoDB(self, data: List[dict], is_train_split: bool=True):
        collection_name = f"{'train' if is_train_split else 'test'}_pairs"
        collection = self.mongoDB.db[collection_name]
        self.mongoDB.insert_documents(collection=collection, insert_data=data)

    def _save_df2original_mongoDB_collection(self, df: pd.DataFrame, id_col_name: str = "author_id"):
        """
        Persist the processed essay DataFrame to the MongoDB collection
        for original (non-paired) texts.

        Each document is stored with:
          - author identifier,
          - text and metadata,
          - dataset name for later retrieval.

        Duplicate inserts are ignored.
        """
        records = df.to_dict(orient="records")
        docs_to_insert = [
            {
                "author": r.pop(id_col_name),
                **r,
                "dataset": self.name,
            }
            for r in records
        ]

        if docs_to_insert:
            try:
                result = self.mongoDB.original_collection.insert_many(
                    docs_to_insert, ordered=False
                )
                logger.info(
                    f"Inserted {len(result.inserted_ids)} documents into '{self.name}' collection.'"
                )
            except errors.BulkWriteError:
                logger.warning(
                    "Duplicate key error encountered during insertMany. Some documents may already exist."
                )
        else:
            logger.info("No new documents to insert.")

    def delete_dataset_from_mongoDB(self):
        mongodb = ParaphraseMongoDB()
        for collection in [
            mongodb.original_collection,
            mongodb.train_pairs_collection,
            mongodb.test_pairs_collection,
        ]:
            n_deleted = mongodb.delete_documents_by_non_id_field(
                collection=collection,
                document_field_name="dataset",
                document_value=self.name,
            )
            logger.info(
                f"Deleted {n_deleted} documents from collection {collection.name} for dataset {self.name}."
            )

    def _return_existing_original_mongodb_collection(self):
        """
        Return already existing MongoDB collection if exists.
        """
        cursor = self.mongoDB.find_document_by_non_id_field(collection=self.mongoDB.original_collection,
                                                            document_field_name="dataset", document_value=self.name)
        records = list(cursor)
        if records and len(records) > 0:
            logger.info(f"Found {len(records)} documents in {CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION}")
            return pd.DataFrame(records)
        else:
            logger.warning(f"No existing documents found in {CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION} for dataset {self.name}.")
            return pd.DataFrame()

    def _make_pair_dict(self, left_text: dict, right_text: dict, same: bool = True):
        """
        Construct a standardized pair dictionary for HuggingFace datasets.

        :param left_text: First essay record.
        :param right_text: Second essay record.
        :param same: Whether both essays originate from the same author.
        :return: Dictionary representing a labeled text pair.
        """
        FEATURE_MAP = {
            "_id": "_id",
            "_author": AUTHOR_COL_NAME,
            "_assignment": ASSIGNMENT_COL_NAME,
        }
        return {
            "dataset_name": self.name,
            **{f"left{key}": left_text[val] for key, val in FEATURE_MAP.items()},
            **{f"right{key}": right_text[val] for key, val in FEATURE_MAP.items()},
            "same": same,
        }
