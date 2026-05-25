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

import logging
import os
from abc import ABC
from datetime import datetime
from typing import Optional

import nltk
from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

nltk.download("punkt_tab")

logger = logging.getLogger(__name__)


class Paraphraser(ABC):
    """
    Abstract base class for paraphrasing models.
    """
    def __init__(self, n_paraphrases: int, model_id: str = CONFIG.OPENAI_MODEL):
        self.n_paraphrases = n_paraphrases
        self.mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
        self.original_collection = self.mongoDB.original_collection
        self.paraphrase_collection = self.mongoDB.non_naive_paraphrase_collection
        self.model_id = model_id

    def paraphrase(
        self, text: str, prompt: str, max_length: int = CONFIG.MAX_LENGTH
    ) -> str:
        """
        Generate a paraphrase of the input text.
        Generate a paraphrase of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A paraphrased version of the input text.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def save_paraphrase_in_mongodb(self, original_text_id:ObjectId, original_text:str, paraphrased_text:str,
                                   extracted_info:Optional[dict], total_costs:Optional[float],
                                   collection, temperature:float=1.0, dataset_name:Optional[str]=None,
                                   prompt:str="bullet points dspy", intermediate_prompt:str="bullet points dspy",) -> None:
        """
        Save the paraphrase to the MongoDB database collection called "paraphrase".
        :param original_text_id: The id of the original text in the original_text collection.
        :param original_text: The original text as a string.
        :param paraphrased_text: The paraphrased text as a string.
        :param extracted_info: A dictionary containing the extracted information from the paraphrase.
        :param collection: The collection where the paraphrase should be stored.
        :param total_costs: The total cost of the paraphrase (including extracted information and paraphrase generation).
        :return:-
        """
        assert (
            type(paraphrased_text) == str
        ), f"paraphrased_text must be a str, but is of type {type(paraphrased_text)}"
        assert len(paraphrased_text.split()) > 0, "paraphrased_text is empty"

        # Build the new document
        paraphrase_doc = {
            # Do not use text_id as _id since a text will be paraphrased multiple times with different settings
            "text_id": ObjectId(original_text_id),  # ID of the original text document
            "length_original_text": len(original_text.split()),
            "length_paraphrased_text": len(paraphrased_text.split()),
            "intermediate_prompt": intermediate_prompt,
            "prompt": prompt,
            "llm": self.model_id,
            "temperature": temperature,
            "paraphrase": paraphrased_text,
            "extracted_info": extracted_info,
            "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "openai_costs": total_costs,
            "dataset_name": dataset_name,
        }

        # Insert into document into paraphrase collection
        collection.insert_one(paraphrase_doc)
        logging.info(f"Inserted paraphrase for document ID: {original_text_id}")
